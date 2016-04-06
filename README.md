# NCon

[![Build Status](https://travis-ci.org/mhauru/NCon.jl.svg?branch=master)](https://travis-ci.org/mhauru/NCon.jl)

NCon provides the module NCon, which exports one function: ncon.
It provides a convenient interface for contracting networks of tensors in a given order.

NCon is a Julia port of the MATLAB function described in [arXiv:1402.0939](https://arxiv.org/abs/1402.0939). NCon relies on the [TensorOperations](https://github.com/Jutho/TensorOperations.jl) package for implementation of pair-wise tensor contractions.
