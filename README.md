# Intensity-Free Learning of Temporal Point Processes

Pytorch implementation of the paper ["Intensity-Free Learning of Temporal Point Processes"](arxiv.org).

## Usage
In order to run the code, you need to install the `dpp` library that contains all the algorithms described in the paper
```bash
cd code
python setup.py install
```

A Jupyter notebook `code/interactive.ipynb` contains the code for training models on the datasets used in the paper.

The same code can also be run as a Python script `code/train.py`.

## Requirements
```
numpy=1.16.4
pytorch=1.2.0
scikit-learn=0.21.2
scipy=1.3.1
```


## Cite
Please cite our paper if you the code or datasets in your own work
```
@article{
    shchur2019inensity,
    title={Intensity-Free Learning of Temporal Point Processes},
    author={Oleksandr Shchur and Marin Bilo\v{s} and Stephan G\"{u}nnemann},
    journal={arXiv preprint},
    year={2019},
}
```
