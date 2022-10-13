# Guided Multi View Stereo

## Installation

Install the dependencies using Conda:

```bash
$ conda env create -f environment.yml
$ conda activate guided-MVS
```

## Download the dataset(s)

Download the pre-processed DTU dataset:

* [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36)
* [MVS Data](http://roboimagedata.compute.dtu.dk/?page_id=36)
* [training set](https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35)
* [evaluation set](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing)

Download Blended MVS, MVS+ and MVS++ from [here](https://github.com/YoYo000/BlendedMVS).

And organize them as follows:
 
```
data/dtu
|-- SampleSet
    |-- MVS Data
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

## Test everything is fine (optional but useful)

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

To train a model, edit the ``params.yaml`` file specifying the model to be trained among the following:

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