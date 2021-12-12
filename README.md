# CompressedSensing.jl
[![CI](https://github.com/SebastianAment/CompressedSensing.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SebastianAment/CompressedSensing.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/SebastianAment/CompressedSensing.jl/branch/main/graph/badge.svg?token=NPYC21MIQT)](https://codecov.io/gh/SebastianAment/CompressedSensing.jl)

Contains a wide-ranging collection of compressed sensing and feature selection algorithms.
Examples include matching pursuit algorithms, forward and backward stepwise regression, sparse Bayesian learning, and basis pursuit.

## Matching Pursuit

The package contains implementations of Matching Pursuit (MP), Orthogonal Matching Pursuit (OMP), and [Generalized OMP](https://arxiv.org/pdf/1111.6664.pdf) (GOMP),
all three of which take advantage of the efficient updating algorithms contained in [UpdatableQRFactorizations.jl](https://github.com/SebastianAment/UpdatableQRFactorizations.jl) to compute the QR factorization of the atoms in the active set.

## Stepwise Regression

## Two-Stage Algorithms

## Sparse Bayesian Learning

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
