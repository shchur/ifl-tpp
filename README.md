# Intensity-Free Learning of Temporal Point Processes

Pytorch implementation of the paper ["Intensity-Free Learning of Temporal Point Processes"](https://openreview.net/forum?id=HygOjhEYDH), Oleksandr Shchur, Marin Biloš and Stephan Günnemann, ICLR 2020.

## Refactored code

The `master` branch contains a refactored version of the code. Some of the original functionality is missing, but the code is much cleaner and should be easier to extend.

You can find the original code (used for experiments in the paper) on branch [`original-code`](https://github.com/shchur/ifl-tpp/tree/original-code).

## Usage
In order to run the code, you need to install the `dpp` library that contains all the algorithms described in the paper
```bash
cd code
python setup.py install
```

A Jupyter notebook [`code/interactive.ipynb`](https://github.com/shchur/ifl-tpp/blob/refactor/code/interactive.ipynb) contains the code for training models on the datasets used in the paper.
The same code can also be run as a Python script [`code/train.py`](https://github.com/shchur/ifl-tpp/blob/refactor/code/train.py).

## Using your own data
You can save your custom dataset in the format used in our code as follows:

```python
dataset = {
    "sequences": [
        {"arrival_times": [0.2, 4.5, 9.1], "marks": [1, 0, 4], "t_start": 0.0, "t_end": 10.0},
        {"arrival_times": [2.3, 3.3, 5.5, 8.15], "marks": [4, 3, 2, 2], "t_start": 0.0, "t_end": 10.0},
    ],
    "num_marks": 5,
}
torch.save(dataset, "data/my_dataset.pkl")
```

## Defining new models
[RecurrentTPP](https://github.com/shchur/ifl-tpp/blob/refactor/code/dpp/models/recurrent_tpp.py) is the base class for marked TPP models.

You just need to inherit from it and implement the `get_inter_time_dist` method that defines how to obtain the distribution (an instance of [`torch.distributions.Distribution`](https://github.com/pytorch/pytorch/blob/master/torch/distributions/distribution.py)) over the inter-event times given the context vector. For example, have a look at the [LogNormMix model](https://github.com/shchur/ifl-tpp/blob/refactor/code/dpp/models/log_norm_mix.py) from our paper.
You can also change the `get_features` and `get_context` methods of `RecurrentTPP` to, for example, use a transformer instead of an RNN.


## Mistakes in the old version
- In the old code we used to normalize the NLL of each sequence by the number of events --- this was incorrect. When computing NLL for multiple TPP sequences, we are only allowed to divide the NLL by the same number for each sequence.
- In the old code we didn't include the survival time of the last event (i.e. time from the last event until the end of the obseved interval) into the NLL computation. This is fixed in the refactored version (and by the way, this seems to be a common mistake in other TPP implementations online).


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
