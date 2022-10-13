import argparse
import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import urllib3
import yaml
from plyfile import PlyData, PlyElement
from torch.utils.data import DataLoader
from tqdm import tqdm

import lib.models as models
from lib.datasets import find_dataset_def, find_scans
from lib.datasets.dtu_utils import read_pfm, save_pfm
from lib.datasets.sample_preprocess import MVSSampleTransform
from lib.utils import *


def main():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    urllib3.disable_warnings()
    cudnn.benchmark = True

    # load args
    parser = argparse.ArgumentParser(description="Predict depth, filter, and fuse")

    # evaluated experiment info
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLFlow run id, if not specified it is retrieved from .current_run.yaml",
    )

    # evaluation params
    parser.add_argument(
        "--dataset",
        type=str,
        default="dtu_yao",
        choices=["dtu_yao", "blended_mvs", "blended_mvg", "eth3d"],
        help="dataset to load",
    )
    parser.add_argument(
        "--hints",
        type=str,
        default=None,
        choices=["not_guided", "guided", "mvguided", "mvguided_filtered"],
        help="if apply hints to the model, integrating from multiple views or not, if not specified is retrieved from mlflow",
    )
    parser.add_argument(
        "--hints-density",
        type=float,
        default=None,
        help="hints density, if not specified is retrieved from mlflow",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=None,
        help="number of views for each sample, if not specified is retrieved from mlflow",
    )

    # for debug or to obtain artifacts
    parser.add_argument(
        "--limit-scans",
        type=int,
        default=None,
        help="limits the number of scans processed, to debug",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        nargs="+",
        default=[],
        choices=[1, 2, 3],
        help="skips step 1, step 2 or step 3, if used it assumes artifacts under ./output. used to debug",
    )
    parser.add_argument(
        "--save-output", action="store_true", help="saves the evaluation artifacts in a ./output"
    )
    parser.add_argument("--debug", action="store_true", help="enables debug logs")

    # load cmd line and params from mlflow
    client = mlflow.tracking.MlflowClient()
    cmd_line_args = parser.parse_args()

    try:
        if cmd_line_args.run_id:
            run = client.get_run(cmd_line_args.run_id)
        else:
            print("searching stored run_uuid")
            with open(".current_run.yaml", "rt") as f:
                run = client.get_run(yaml.safe_load(f)["run_uuid"])

        exp_name = client.get_experiment(run.info.experiment_id).name
        run_name = run.data.tags.get("mlflow.runName", run.info.run_uuid)
        print(f"evaluating exp: {exp_name}, run: {run_name}")
    except FileNotFoundError:
        print(
            "Error: current run not defined, please launch a training or specify --experiment_id and --run_id"
        )
        return 1
    except Exception as e:
        print("Error:", str(e))
        return 2

    args = SimpleNamespace(**vars(cmd_line_args))
    args.model = run.data.params.pop("model")
    for k, v in run.data.params.items():
        if k in [
            "epochs",
            "steps",
            "batch_size",
            "epochs_lr_decay",
            "epochs_lr_gamma",
            "ndepths",
            "views",
        ]:
            if v != "None":
                run.data.params[k] = int(v)
            else:
                run.data.params[k] = None
        if k in ["lr", "weight_decay", "hints_density"]:
            if v is not None:
                run.data.params[k] = float(v)
        if k == "hints_filter_window":
            h_window, w_window = re.compile(r"\[(\d),\s+(\d)\]").match(v).groups()
            run.data.params[k] = (int(h_window), int(w_window))

    args.train = SimpleNamespace(**run.data.params)

    # debug
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # output folder
    if args.skip_steps and not args.save_output:
        print("Error: when using --skip-steps --save-output must be used too")
        return 3

    if args.save_output:
        output_folder = Path("eval_output")
        print(f"results stored in {output_folder}")
    else:
        output_folder = Path(tempfile.mkdtemp())
        print(f"temporary results in {output_folder}")

    # seed everything
    pl.seed_everything(42)

    # data
    if args.hints is None:
        args.hints = args.train.hints
    if args.views is None:
        args.views = args.train.views
    if args.hints_density is None:
        args.hints_density = args.train.hints_density

    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(
        "test",
        nviews=args.views,
        ndepths=args.train.ndepths,
        transform=MVSSampleTransform(
            generate_hints=args.hints,
            hints_perc=args.hints_density,
            filtering_window=tuple(args.train.hints_filter_window),
        ),
    )

    if args.limit_scans is not None:
        metas = test_dataset.metas
        scans = sorted(list(set(m[0] for m in metas)))[: args.limit_scans]
        test_dataset.metas = [m for m in metas if m[0] in scans]
    else:
        scans = find_scans(args.dataset, "test")

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )

    # model
    model = models.build_network(args.model, args)
    model = nn.DataParallel(model)
    model.cuda()

    print(f"loading model from run {run_name}")
    output_folder.mkdir(exist_ok=True, parents=True)
    model_path = client.download_artifacts(run.info.run_uuid, "model.ckpt", output_folder)

    state_dict = torch.load(model_path)["state_dict"]
    model.module.model.load_state_dict(
        {".".join(k.split(".")[2:]): v for k, v in state_dict.items()}
    )
    model.eval()

    # step 1
    # for each testing scene is computed and saved the depth and confidence of each frame
    if 1 not in args.skip_steps:
        with torch.no_grad():
            for sample in tqdm(test_dataloader, desc="Step 1 saving depth and confidence"):

                sample_cuda = tocuda(sample)
                outputs = tensor2numpy(model(sample_cuda))
                filenames = sample["filename"]

                for filename, depth_est, photometric_confidence in zip(
                    filenames, outputs["depth"]["stage_0"], outputs["photometric_confidence"]
                ):
                    depth_filename: Path = output_folder / filename.format("depth_est", ".pfm")
                    confidence_filename: Path = output_folder / filename.format(
                        "confidence", ".pfm"
                    )
                    depth_filename.parent.mkdir(parents=True, exist_ok=True)
                    confidence_filename.parent.mkdir(parents=True, exist_ok=True)
                    depth_est = np.squeeze(depth_est, 0)
                    save_pfm(str(depth_filename), depth_est)
                    save_pfm(str(confidence_filename), photometric_confidence)
    else:
        print("skipped step 1")

    # step 2
    # for each scan saves on disk the final .ply point cloud
    if 2 not in args.skip_steps:
        if args.dataset == "dtu_yao":
            for scan in tqdm(scans, desc="Step 2 computing complete point clouds"):
                scan_id = int(scan[4:])
                scan_folder: Path = Path("data/dtu/test_data") / scan
                out_folder: Path = output_folder / scan
                save_final_ply(
                    str(scan_folder),
                    str(out_folder),
                    str(output_folder / "pcd_scan_{:0>3}.ply".format(scan_id)),
                    1.0,  # geo_pixel_thres
                    0.01,  # geo_depth_thres
                    0.8,  # photo_thres
                )
        else:
            print(f"step 2 not enabled for dataset {args.dataset}")
    else:
        print("skipped step 2")

    # step 3
    # compute the final metrics
    if 3 not in args.skip_steps:
        metrics = defaultdict(lambda: [])
        scans_processed = []
        for sample in tqdm(test_dataloader, desc="Step 3 computing metrics"):

            if args.dataset == "dtu_yao":
                scan = int(sample["filename"][0].split("/")[0][4:])
            else:
                scan = sample["filename"][0]

            # compute 3D metrics for DTU once for each scan
            if args.dataset == "dtu_yao" and scan not in scans_processed:
                scans_processed.append(scan)

                # load data
                gt_pcd = sample["scan_pcd"][0].numpy()
                bounding_box = sample["scan_pcd_bounding_box"][0].numpy()
                obs_mask = sample["scan_pcd_obs_mask"][0].numpy()
                ground_plane = sample["scan_pcd_ground_plane"][0].numpy()
                res = sample["scan_pcd_resolution"][0].numpy()

                mesh = PlyData.read(output_folder / "pcd_scan_{:0>3}.ply".format(scan))
                pred_pcd = np.stack(
                    [mesh["vertex"]["x"], mesh["vertex"]["y"], mesh["vertex"]["z"]], -1
                )

                # compute distances
                pred_gt_distance = pcd_distances(pred_pcd, gt_pcd, bounding_box, 60)
                gt_pred_distance = pcd_distances(gt_pcd, pred_pcd, bounding_box, 60)

                # filter pred-gt distances
                normalized_pcd = (np.rint(pred_pcd - bounding_box[0:1]) / res).astype(int)
                valid1 = (normalized_pcd >= 0).all(axis=1) & (
                    normalized_pcd < np.array(obs_mask.shape)[None]
                ).all(axis=1)
                normalized_pcd = normalized_pcd[valid1]
                valid_mask = np.zeros((pred_pcd.shape[0],), dtype=np.bool)
                valid2 = obs_mask.astype(bool)[
                    normalized_pcd[:, 0], normalized_pcd[:, 1], normalized_pcd[:, 2]
                ]
                valid_mask[np.where(valid1)[0][valid2]] = True
                pred_gt_distance = pred_gt_distance[valid_mask & (pred_gt_distance < 20)]

                # filter gt-pred distances
                above_plane = (
                    np.matmul(
                        np.concatenate([gt_pcd, np.ones([len(gt_pcd), 1])], axis=-1),
                        ground_plane,
                    ).flatten()
                    > 0
                )
                gt_pred_distance = gt_pred_distance[above_plane & (gt_pred_distance < 20)]

                metrics["accuracy"].append(pred_gt_distance.mean())
                metrics["completeness"].append(gt_pred_distance.mean())

            # compute 2D metrics for all datasets
            depth_gt = sample["depth"]["stage_0"].numpy()
            mask = (depth_gt > 0).astype(depth_gt.dtype)
            depth_est = read_pfm(
                output_folder / sample["filename"][0].format("depth_est", ".pfm")
            )[0].transpose([2, 0, 1])[None]
            for thresh in [1, 2, 3, 4, 8]:
                metrics[f"perc_l1_upper_thresh_{thresh}"].append(
                    np.mean(np.abs(depth_est[mask > 0.5] - depth_gt[mask > 0.5]) > thresh)
                )
            metrics["mae"].append(np.mean(np.abs(depth_est[mask > 0.5] - depth_gt[mask > 0.5])))

        metrics = {k: np.stack(v).mean().item() for k, v in metrics.items()}
        for k, v in metrics.items():
            metric_name = f"test-{args.dataset}"
            if args.hints == "not_guided":
                metric_name = metric_name + f"/{args.hints}/views_{args.views}/{k}"
            else:
                metric_name = (
                    metric_name + f"/{args.hints}-{args.hints_density}/views_{args.views}/{k}"
                )
            client.log_metric(run.info.run_uuid, metric_name, v)

        print("METRICS")
        for k, v in metrics.items():
            print("{:>20}       {:.5f}".format(k, v))
    else:
        print("skipped step 3")

    if not args.save_output:
        print(f"removing temp folder {output_folder}")
        shutil.rmtree(output_folder, ignore_errors=True)


def pad_slice(pad):
    if pad == 0:
        return slice(0, None)
    elif pad % 2 == 0:
        return slice(pad // 2, -pad // 2)
    else:
        pad_1 = pad // 2
        pad_2 = (pad // 2) + 1
        return slice(pad_1, -pad_2)


def save_final_ply(
    scan_folder,
    out_folder,
    plyfilename,
    geo_pixel_thres,
    geo_depth_thres,
    photo_thres,
):
    """
    Saves the final RGB .ply file for each scan in output/eval_<dataset>/<scan>/
    """
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs, vertex_colors = None, None
    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    pbar = tqdm(total=len(pair_data) + 2, leave=False)
    for ref_view, src_views in pair_data:

        pbar.set_description(
            "ref view: " + str(ref_view) + " - src views: " + ", ".join(str(x) for x in src_views)
        )

        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, "cams_1/{:0>8}_cam.txt".format(ref_view))
        )
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, "images/{:0>8}.jpg".format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(
            os.path.join(out_folder, "depth_est/{:0>8}.pfm".format(ref_view))
        )[0]
        ref_depth_est = np.squeeze(ref_depth_est, 2)
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, "confidence/{:0>8}.pfm".format(ref_view)))[
            0
        ]

        # (remove padding from the dataloader)
        h_img, w_img, _ = ref_img.shape
        h_est, w_est, _ = confidence.shape
        h_slice = pad_slice(h_est - h_img)
        w_slice = pad_slice(w_est - w_img)
        ref_depth_est = ref_depth_est[h_slice, w_slice]
        confidence = confidence[h_slice, w_slice]

        photo_mask = confidence > photo_thres
        photo_mask = np.squeeze(photo_mask, 2)

        all_srcview_depth_ests = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, "cams_1/{:0>8}_cam.txt".format(src_view))
            )
            # the estimated depth of the source view
            src_depth_est = read_pfm(
                os.path.join(out_folder, "depth_est/{:0>8}.pfm".format(src_view))
            )[0]

            # (remove padding from the dataloader)
            h_est, w_est, _ = src_depth_est.shape
            h_slice = pad_slice(h_est - h_img)
            w_slice = pad_slice(w_est - w_img)
            src_depth_est = src_depth_est[h_slice, w_slice]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics,
                src_extrinsics,
                geo_pixel_thres,
                geo_depth_thres,
            )
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)

        # at least 3 source views matched
        # large threshold, high accuracy, low completeness
        geo_mask = geo_mask_sum >= 3
        final_mask = np.logical_and(photo_mask, geo_mask)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        valid_points = final_mask
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        color = ref_img[valid_points]
        xyz_ref = np.matmul(
            np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth
        )
        xyz_world = np.matmul(
            np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x)))
        )[:3]

        if vertexs is None and vertex_colors is None:
            vertexs = xyz_world.transpose([1, 0])
            vertex_colors = (color * 255).astype(np.uint8)
        else:
            vertexs = np.concatenate([vertexs, xyz_world.transpose([1, 0])], axis=0)
            vertex_colors = np.concatenate(
                [
                    vertex_colors,
                    (color * 255).astype(np.uint8),
                ],
                axis=0,
            )

        pbar.update(1)

    pbar.set_description("pruning final point cloud")
    mask = prune_pcd(vertexs, 0.2, n_jobs=4)
    pbar.update(1)

    pbar.set_description("save pruned point cloud")
    vertexs, vertex_colors = vertexs[mask], vertex_colors[mask]
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_colors = np.array(
        [tuple(v) for v in vertex_colors], dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(plyfilename)
    pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
