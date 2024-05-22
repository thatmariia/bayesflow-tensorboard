# Template for BayesFlow Trainer + TensorBoard

This repository demonstrates how to integrate [TensorBoard](https://github.com/tensorflow/tensorboard) with [BayesFlow](https://github.com/stefanradev93/BayesFlow) Trainer. 
BayesFlow is a framework for training Bayesian neural networks, 
but it doesn't natively support TensorBoard, which is a popular tool for visualizing training progress and metrics. 
This minimal example / template provides a minimal implementation to bridge that gap.

## Usage

Before running the code, make sure you have installed Python 3.10 (other versions may work) and the required packages.

* Install the packages: `pip install -r requirements.txt`
* Run the code: `python3 -msrc.main`
* Run `tensorboard --logdir logs/fit` and open the browser at `localhost:6006`
