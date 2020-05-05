# Intensity-Free Learning of Temporal Point Processes

Pytorch implementation of the paper ["Intensity-Free Learning of Temporal Point Processes"](https://openreview.net/forum?id=HygOjhEYDH), Oleksandr Shchur, Marin Biloš and Stephan Günnemann, ICLR 2020.

## Usage
In order to run the code, you need to install the `dpp` library that contains all the algorithms described in the paper
```bash
cd code
python setup.py install
```

A Jupyter notebook [`code/interactive.ipynb`](https://github.com/shchur/ifl-tpp/blob/master/code/interactive.ipynb) contains the code for training models on the datasets used in the paper.
Another notebook [`code/generate_embeddings.ipynb`](https://github.com/shchur/ifl-tpp/blob/master/code/generate_embeddings.ipynb) shows how to learn sequence embeddings for different synthetic datasets.

The same code can also be run as a Python script `code/train.py`.

## Requirements
```
numpy=1.16.4
pytorch=1.2.0
scikit-learn=0.21.2
scipy=1.3.1
```


## Cite
Please cite our paper if you use the code or datasets in your own work
```
@article{
    shchur2020intensity,
    title={Intensity-Free Learning of Temporal Point Processes},
    author={Oleksandr Shchur and Marin Bilo\v{s} and Stephan G\"{u}nnemann},
    journal={International Conference on Learning Representations (ICLR)},
    year={2020},
}
```
