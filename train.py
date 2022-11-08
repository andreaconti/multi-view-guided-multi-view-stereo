import argparse
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytorch_lightning as pl
import urllib3
import yaml
from git import Repo
from git.exc import InvalidGitRepositoryError
from mlflow.tracking import MlflowClient
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import guided_mvs_lib.models as models
from guided_mvs_lib.datasets import MVSDataModule
from guided_mvs_lib.datasets.sample_preprocess import MVSSampleTransform
from guided_mvs_lib.utils import *


def run_training(
    params: Union[str, Path, dict] = "params.yaml",
    cmdline_args: Optional[List[str]] = None,
    datapath: Union[str, Path, None] = None,
    outpath: Union[str, Path] = "output",
    logspath: Union[str, Path] = ".",
):
    # handle args
    outpath = Path(outpath)
    logspath = Path(logspath)

    # remove annoying torch specific version warnings
    warnings.simplefilter("ignore", UserWarning)
    urllib3.disable_warnings()

    parser = argparse.ArgumentParser(description="training procedure")

    # training params
    parser.add_argument(
        "--gpus", type=int, default=1, help="number of gpus to select for training"
    )
    parser.add_argument(
        "--fast-dev-run",
        nargs="?",
        const=True,
        default=False,
        help="if execute a single step of train and val, to debug",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="limits the number of batches for each epoch, to debug",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="limits the number of batches for each epoch, to debug",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="if resume from the last checkpoint or from a specific checkpoint",
    )
    parser.add_argument(
        "--load-weights",
        default=None,
        type=str,
        help="load weights either from a mlflow train or from a checkpoint file",
    )

    # experiment date
    date = datetime.now().strftime(r"%Y-%h-%d-%H-%M")

    # parse arguments and merge from params.yaml
    cmd_line_args = parser.parse_args(cmdline_args)
    if isinstance(params, dict):
        train_args = params
    else:
        with open(params, "rt") as f:
            train_args = yaml.safe_load(f)

    args = SimpleNamespace(**vars(cmd_line_args))
    for k, v in train_args.items():
        if not isinstance(v, dict):
            setattr(args, k, v)
        else:
            setattr(args, k, SimpleNamespace(**v))

    # Train using pytorch lightning
    pl.seed_everything(42)

    # Build LightningDataModule
    data_module = MVSDataModule(
        args.train.dataset,
        batch_size=args.train.batch_size,
        datapath=datapath,
        nviews=args.train.views,
        ndepths=args.train.ndepths,
        robust_train=True if args.train.dataset == "dtu_yao" else False,
        transform=MVSSampleTransform(
            generate_hints=args.train.hints,
            hints_perc=args.train.hints_density,
            filtering_window=tuple(args.train.hints_filter_window),
        ),
    )

    # loading model or only weights ?
    if args.load_weights is not None and args.resume_from_checkpoint is not False:
        print("Use either --load-weights or --resume-from-checkpoint")
        return

    ckpt_path = None
    steps_re = re.compile("step=(\d+)")
    if args.resume_from_checkpoint is True:
        if (outpath / "ckpts/last.ckpt").exists():
            ckpt_path = outpath / "ckpts/last.ckpt"
        else:
            ckpts = list((outpath / "ckpts").glob("*.ckpt"))
            steps = [
                int(steps_re.findall(ckpt.name)[0])
                for ckpt in ckpts
                if steps_re.findall(ckpt.name) is not []
            ]
            if not steps:
                print("not found any valid checkpoint in", str(outpath / "ckpts"))
                return
            ckpt_path = ckpts[np.argmax(steps)]
        print(f"resuming from last checkpoint: {ckpt_path}")
    elif args.resume_from_checkpoint is not False:
        if Path(args.resume_from_checkpoint).exists():
            ckpt_path = args.resume_from_checkpoint
            print(f"resuming from choosen checkpoint: {ckpt_path}")
        else:
            print(f"file {ckpt_path} does not exist")
            return

    # init mlflow logger and model
    if ckpt_path is None:
        logger = MLFlowLogger(
            experiment_name="guided-mvs",
            run_name=f"{args.model}-{date}",
        )

        outpath.mkdir(exist_ok=True, parents=True)
        with open(outpath / "run_uuid", "wt") as f:
            f.write(logger.run_id)

        model = models.MVSModel(
            args=args,
            mlflow_run_id=logger.run_id,
            v_num=f"{args.model}-{'-'.join(date.split('-')[1:3])}",
        )
    else:

        with open(outpath / "run_uuid", "rt") as f:
            mlflow_run_id = f.readline().strip()

        model = models.MVSModel.load_from_checkpoint(
            ckpt_path,
            args=args,
            mlflow_run_id=mlflow_run_id,
            v_num=f"{args.model}-{'-'.join(date.split('-')[1:3])}",
        )
        logger = MLFlowLogger(
            experiment_name="guided-mvs",
            run_name=f"{args.model}-{date}",
        )
        logger._run_id = mlflow_run_id

    # if required load weights
    if args.load_weights is not None:
        mlflow_client: MlflowClient = logger.experiment
        if args.load_weights in [
            run.run_uuid for run in mlflow_client.list_run_infos(logger.experiment_id)
        ]:
            # download the model
            run_weights_path = mlflow_client.download_artifacts(args.load_weights, "model.ckpt")
            model.load_state_dict(torch.load(run_weights_path)["state_dict"])

            # track the model weights
            run_weights_path = Path(run_weights_path)
            shutil.move(run_weights_path, run_weights_path.parent / "init_weights.ckpt")
            mlflow_client.log_artifact(
                logger.run_id, run_weights_path.parent / "init_weights.ckpt"
            )
            mlflow_client.set_tag(logger.run_id, "load_weights", args.load_weights)
            shutil.rmtree(Path(run_weights_path).parent, ignore_errors=True)
        else:
            try:
                model.load_state_dict(torch.load(args.load_weights)["state_dict"])
                tmpdir = Path(tempfile.mkdtemp())
                shutil.copy(args.load_weights, tmpdir / "init_weights.ckpt")
                mlflow_client.log_artifact(logger.run_id, tmpdir / "init_weights.ckpt")
                shutil.rmtree(tmpdir, ignore_errors=True)
            except FileNotFoundError:
                print(f"{args.load_weights} is neither a valid run id or a path to a .ckpt")
                return

    # handle checkpoints
    if (
        args.train.epochs is None
        or args.train.epochs == 1
        and args.train.steps is not None
        and args.train.steps > 0
    ):
        ckpt_callback = ModelCheckpoint(
            outpath / "ckpts",
            train_time_interval=timedelta(hours=2),
            save_last=True,
        )
    else:
        ckpt_callback = ModelCheckpoint(outpath / "ckpts", save_last=True)

    remove_output = True

    class HandleOutputs(Callback):
        def on_train_end(self, trainer, pl_module):

            # save final model
            print("saving the final model.")
            torch.save(
                {"global_step": trainer.global_step, "state_dict": pl_module.state_dict()},
                outpath / "model.ckpt",
            )

            # copy the model and the params on MLFlow
            if not args.fast_dev_run:
                mlflow_client: MlflowClient = logger.experiment

                # store diff file if needed
                try:
                    repo = Repo(Path.cwd())

                    if repo.is_dirty():
                        try:
                            out = subprocess.check_output(["git", "diff"], cwd=Path.cwd())
                            if out is not None:
                                tmpfile = Path(tempfile.mkdtemp()) / "changes.diff"
                                with open(tmpfile, "wb") as f:
                                    f.write(out)
                                mlflow_client.log_artifact(logger.run_id, tmpfile)
                                os.remove(tmpfile)
                        except subprocess.CalledProcessError as e:
                            print("Failed to save a diff file of the current experiment")

                except InvalidGitRepositoryError:
                    pass

                # save the model
                mlflow_client.log_artifact(logger.run_id, str(outpath / "model.ckpt"))

                # finally, remove the temp output and log in a hidden file the current run
                # for the eval step
                with open(".current_run.yaml", "wt") as f:
                    yaml.safe_dump(
                        {"experiment": logger.experiment_id, "run_uuid": logger.run_id}, f
                    )

        def on_keyboard_interrupt(self, trainer, pl_module):
            print("training interrupted")

            # (not removing checkpoints)
            nonlocal remove_output
            remove_output = False

    # init train
    trainer_params = {
        "gpus": args.gpus,
        "fast_dev_run": args.fast_dev_run,
        "logger": logger,
        "benchmark": True,
        "callbacks": [ckpt_callback, HandleOutputs()],
        "weights_summary": None,
        "resume_from_checkpoint": ckpt_path,
        "num_sanity_val_steps": 0,
    }

    if (
        args.resume_from_checkpoint is not False
        and args.train.epochs is not None
        and args.train.epochs == 1
        and args.train.steps is not None
        and args.train.steps > 0
        and ckpt_path is not None
    ):
        args.train.epochs = None

    if args.train.epochs is not None:
        trainer_params["max_epochs"] = args.train.epochs
    if args.train.steps is not None:
        trainer_params["max_steps"] = args.train.steps
    if args.limit_train_batches is not None:
        trainer_params["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_params["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model, data_module)

    if remove_output:
        shutil.rmtree(outpath)


if __name__ == "__main__":
    run_training()
