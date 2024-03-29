# Multi-View Guided Multi-View Stereo

<p>
<div align="center">
    <a href="https://mattpoggi.github.io">Matteo Poggi</a>*
    &middot;
    <a href="https://andreaconti.github.io">Andrea Conti</a>*
    &middot;
    <a href="http://vision.deis.unibo.it/~smatt/Site/Home.html">Stefano Mattoccia</a>
    <i>*joint first authorship</i>
</div>
<div align="center">
    <a href="https://arxiv.org/pdf/2210.11467v1.pdf">[Arxiv]</a>
    <a href="https://andreaconti.github.io/projects/multiview_guided_multiview_stereo/">[Project Page]</a>
    <a href="https://github.com/andreaconti/multi-view-guided-multi-view-stereo/blob/main/results.ipynb">[Demo]</a>
</div>
</p>

This is the official source code of Multi-View Guided Multi-View Stereo presented at [IEEE/RSJ International Conference on Intelligent Robots and Systems](https://iros2022.org/)

## Citation

```
@inproceedings{poggi2022guided,
  title={Multi-View Guided Multi-View Stereo},
  author={Poggi, Matteo and Conti, Andrea and Mattoccia, Stefano},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems},
  note={IROS},
  year={2022}
}
```

## Load pretrained models and evaluate

We release many of the mvs networks tested in the paper trained on Blended-MVG or on Blended-MVG and fine-tuned on DTU, with and without sparse depth points. To load these models can be simply used the `torch.hub` API.

```python
model = torch.hub.load(
    "andreaconti/multi-view-guided-multi-view-stereo",
    "mvsnet",               # mvsnet | ucsnet | d2hc_rmvsnet | patchmatchnet | cas_mvsnet
    pretrained=True,
    dataset="blended_mvg",  # blended_mvg | dtu_yao_blended_mvg
    hints="not_guided",     # mvguided_filtered | not_guided | guided | mvguided
)
```

Once loaded each model have the same following interface, moreover each pretrained models provides its training parameters under the attribute `train_params`.

```python
depth = model(
    images,        # B x N x 3 x H x W
    intrinsics,    # B x N x 3 x 3
    extrinsics,    # B x N x 4 x 4
    depth_values,  # B x D (128 usually)
    hints,         # B x 1 x H x W (optional)
)
```

Finally, we provide also an interface over the datasets used as follows. In this case is required Pytorch Lightning as dependency and the dataset must be stored locally.

```python
dm = torch.hub.load(
    "andreaconti/multi-view-guided-multi-view-stereo",
    "blended-mvg",  # blended_mvg | blended_mvs | dtu
    root="data/blended-mvg",
    hints="not_guided",     # mvguided_filtered | not_guided | guided | mvguided
    hints_density=0.03,
)
dm.prepare_data()
dm.setup()
dl = dm.train_dataloader()
```

In [results.ipynb](https://github.com/andreaconti/multi-view-guided-multi-view-stereo/blob/main/results.ipynb) there is an example of how to reproduce some of the results showed in the paper through the `torch.hub` API.

## Installation

Install the dependencies using Conda or [Mamba](https://github.com/mamba-org/mamba):

```bash
$ conda env create -f environment.yml
$ conda activate guided-mvs
```

## Download the dataset(s)

Download the used datasets:

* [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36) preprocessed by [patchmatchnet](https://github.com/FangjinhuaWang/PatchmatchNet), [train](https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35) and [val](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing)
* [BlendedMVG](https://github.com/YoYo000/BlendedMVS), download the lowres version from [BlendedMVS](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDu7FHfbPZwqhIy?e=BHY07t), [BlendedMVS+](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVLILxpohZLEYiIa?e=MhwYSR), [BlendedMVS++](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVHCxmURGz0UBGns?e=Tnw2KY)

And organize them as follows under the data folder (sym-links works fine):
 
```
data/dtu
|-- train_data
    |-- Cameras_1
    |-- Depths_raw
    |-- Rectified
|-- test_data
    |-- scan1 
    |-- scan2
    |-- ..

data/blended-mvs
|-- <scan hash name>
    |-- blended_images
    |-- cams 
    |-- rendered_depth_maps
|-- ..
```

## [Optional] Test everything is fine

This project implements some tests to preliminarily check everything is fine. Tests are grouped by different tags.

``` 
# tages:
# - data: tests related to the datasets
# - dtu: tests related to the DTU dataset
# - blended_mvs: tests related to blended MVS
# - blended_mvg: tests related to blended MVG
# - train: tests to launch all networks in fast dev run mode (1 batch of train val for each network for each dataset)
# - slow: tests slow to be executed

# EXAMPLES

# runs all tests
$ pytest

# runs tests excluding slow ones
$ pytest -m "not slow"

# runs tests only on dtu
$ pytest -m dtu

# runs tests on data except for dtu ones
$ pytest -m "data and not dtu"
```

## Training

To train a model, edit ``params.yaml`` specifying the model to be trained among the following:

* [cas_mvsnet](https://arxiv.org/pdf/1912.06378.pdf)
* [d2hc_rmvsnet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490647.pdf)
* [mvsnet](https://arxiv.org/pdf/1804.02505.pdf)
* [patchmatchnet](https://arxiv.org/pdf/2012.01411.pdf)
* [ucsnet](https://arxiv.org/abs/1911.12012)

The dataset between ``dtu_yao``, `blended_mvs`, ``blended_mvg`` and the other training parameters, then hit:

```
# python3 train.py --help to see the options
$ python3 train.py
```

The best model is stored in ``output`` folder as ``model.ckpt`` along with a ``meta`` folder containing useful informations about the training executed.

### Resume a training

If something bad happens or if you stop the training process using a keyboard interrupt (``Ctrl-C``) the checkpoints will not be deleted and you can resume
the training with the following option:

```
# resume the last checkpoint saved in output/ckpts (last epoch)
$ python3 train.py --resume-from-checkpoint

# resume a choosen checkpoint elsewere
$ python3 train.py --resume-from-checkpoint my_checkpoint.ckpt
```

It takes care to properly update the correct training logs on tensorboard.

## Evaluation

Once you have trained the model you can evaluate it using ``eval.py``, here you have few options, specifically:

* the **dataset** on which test
* if evaluate using **guided** hints, **guided_integral** hints or none
* the **hints density** to be used
* the number of **views**

```
# see the options
$ python3 eval.py --help
usage: eval.py [-h] [--dataset {dtu_yao,blended_mvs,blended_mvg}]
               [--hints {not_guided,guided,guided_integral}]
               [--hints-density HINTS_DENSITY] [--views VIEWS]
               [--loadckpt LOADCKPT] [--limit-scans LIMIT_SCANS]
               [--skip-steps {1,2,3} [{1,2,3} ...]] [--save-scans-output]

# EXAMPLES
# without guided hints on dtu_yao, 3 views
$ python3 eval.py

# with guided hints and 5 views and density of 0.01
$ python3 eval.py --hints guided --views 5

# with integral guided hints 3 views and 0.03 density
$ python3 eval.py --hints guided_integral --hints-density 0.03
```

Results will be stored under ``output/eval_<dataset>/[guided|not_guided|guided_integral]-[density=<hints-density>]-views=<views>``, for instance guiding on DTU with a 0.01 density and using 3 views the results will be in:

* ``output/eval_dtu_yao/guided-density=0.01-views=3/``

Each of these folders will contain the point cloud for each testing scene and a ``metrics.json`` file containing the final metrics, they will differ depending on the dataset used for evaluation.

## Development

### Environment

To develop you have to create a conda virtual environment and **also** install git hooks:

```bash
$ conda env create -f environment.yml
$ conda activate guided-MVS
$ pre-commit install
```

When you will commit [Black](https://github.com/psf/black) and [Isort](https://pypi.org/project/isort/) will be executed on the modified
files.

### VSCode specific settings

If you use Visual Studio Code its configuration and needed extensions are stored in ``.vscode``. Create a file in the root folder called ``.guided-mvs.code-workspace`` containing the following to load the conda environment properly:

```json
{
    "folders": [
        {
            "path": "."
        }
    ],
    "settings": {
        "python.pythonPath": "<path to your python executable (see in conda)>"
    }
}
```
