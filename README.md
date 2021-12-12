# CompressedSensing.jl
[![CI](https://github.com/SebastianAment/CompressedSensing.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SebastianAment/CompressedSensing.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/SebastianAment/CompressedSensing.jl/branch/main/graph/badge.svg?token=NPYC21MIQT)](https://codecov.io/gh/SebastianAment/CompressedSensing.jl)

Contains a wide-ranging collection of compressed sensing and feature selection algorithms.
Examples include matching pursuit algorithms, forward and backward stepwise regression, sparse Bayesian learning, and basis pursuit.

## Matching Pursuits

The package contains implementations of [Matching Pursuit (MP)](https://en.wikipedia.org/wiki/Matching_pursuit),
[Orthogonal Matching Pursuit (OMP)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=342465),
and [Generalized OMP (GOMP)](https://arxiv.org/pdf/1111.6664.pdf),
all three of which take advantage of the efficient updating algorithms contained in [UpdatableQRFactorizations.jl](https://github.com/SebastianAment/UpdatableQRFactorizations.jl) to compute the QR factorization of the atoms in the active set.

## Stepwise Regression
- Forward Regression
- Backward Regression
- 
## Two-Stage Algorithms

- [Subspace Pursuit (SP)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4839056&casa_token=fyP4fT6vvjAAAAAA:cT_80KeMMH3WycQA0f-HqXUj0hViY-fSajRgENqYmyOhOHWXTq5EIRE5rcpZl675JyHO917Trw).
- Relevance Matching Pursuit (RMP) introduced in [Sparse Bayesian Learning via Stepwise Regression](https://proceedings.mlr.press/v139/ament21a.html).
- Stepwise Regression with Replacement (SRR) introduced in [On the Optimality of Backward Regression: Sparse Recovery and Subset Selection](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9415082&casa_token=gmwN6_yXZSAAAAAA:uzKJOGwZFFwZzum2SoZWNtsvcpSQ34Rdib_0PlyU3oNDY-ZkB9PULGNGGnuHjSC2U51YiywiSQ).


## Sparse Bayesian Learning

- Original SBL algorithm introduced in [Sparse Bayesian Learning and the Relevance Vector Machine](https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf).
- [Fast Marginal Likelihood Maximisation for
Sparse Bayesian Models](http://www.miketipping.com/papers/met-fastsbl.pdf)

## Basis Pursuit

Basis Pursuit (BP) with reweighting schemes, like the ones related to entropy regularization and the Automatic Relevance Determination (ARD) or SBL prior.

## Citing this Package
This package was written in the course of a research project on sparsity-promiting algorithms and was published with the paper [Sparse Bayesian Learning via Stepwise Regression](https://proceedings.mlr.press/v139/ament21a.html).
Consider using the following citation, when referring to this package in a publication.
```bib
@InProceedings{pmlr-v139-ament21a,
  title = 	 {Sparse Bayesian Learning via Stepwise Regression},
  author =       {Ament, Sebastian E. and Gomes, Carla P.},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {264--274},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/ament21a/ament21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/ament21a.html},
  abstract = 	 {Sparse Bayesian Learning (SBL) is a powerful framework for attaining sparsity in probabilistic models. Herein, we propose a coordinate ascent algorithm for SBL termed Relevance Matching Pursuit (RMP) and show that, as its noise variance parameter goes to zero, RMP exhibits a surprising connection to Stepwise Regression. Further, we derive novel guarantees for Stepwise Regression algorithms, which also shed light on RMP. Our guarantees for Forward Regression improve on deterministic and probabilistic results for Orthogonal Matching Pursuit with noise. Our analysis of Backward Regression culminates in a bound on the residual of the optimal solution to the subset selection problem that, if satisfied, guarantees the optimality of the result. To our knowledge, this bound is the first that can be computed in polynomial time and depends chiefly on the smallest singular value of the matrix. We report numerical experiments using a variety of feature selection algorithms. Notably, RMP and its limiting variant are both efficient and maintain strong performance with correlated features.}
}
```
