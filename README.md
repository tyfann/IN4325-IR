# allRank : Learning to Rank in PyTorch

### Motivation

allRank provides an easy and flexible way to experiment with various LTR neural network models and loss functions.
It is easy to add a custom loss, and to configure the model and the training procedure. 
We hope that allRank will facilitate both research in neural LTR and its industrial applications.

## Features

### Implemented loss functions:
 1. RankNet
 2. ApproxNDCG
 3. RMSE
 4. NeuralNDCG (introduced in https://arxiv.org/pdf/2102.07831)

### Getting started guide

Download dataset and put it into data folder(you need to create that one in the root folder)

first run

```shell
pip install -r requirements.txt
```

after that run pipeline_test.


## Research

This framework was developed to support the research project [Context-Aware Learning to Rank with Self-Attention](https://arxiv.org/abs/2005.10084). If you use allRank in your research, please cite:
```
@article{Pobrotyn2020ContextAwareLT,
  title={Context-Aware Learning to Rank with Self-Attention},
  author={Przemyslaw Pobrotyn and Tomasz Bartczak and Mikolaj Synowiec and Radoslaw Bialobrzeski and Jaroslaw Bojar},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.10084}
}
```
Additionally, if you use the NeuralNDCG loss function, please cite the corresponding work, [NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting](https://arxiv.org/abs/2102.07831):
```
@article{Pobrotyn2021NeuralNDCG,
  title={NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting},
  author={Przemyslaw Pobrotyn and Radoslaw Bialobrzeski},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.07831}
}
```

## License

Apache 2 License
