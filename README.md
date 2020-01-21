# NCon

[![Build Status](https://travis-ci.org/mhauru/NCon.jl.svg?branch=master)](https://travis-ci.org/mhauru/NCon.jl)

**UPDATE January 2020**: Since November 2019 [TensorOperations](https://github.com/Jutho/TensorOperations.jl) implements an `ncon` interface as well. It hence provides everything that this package does, plus much more, such as smart management of temporary arrays. We hence recommend using TensorOperations instead of NCon from now on. Future maintenance of NCon may or may not happen.

NCon exports the function `ncon`, which provides a convenient interface for
contracting networks of tensors in a given order.  It is a Julia port of the
MATLAB function described in
[arXiv:1402.0939](https://arxiv.org/abs/1402.0939), although without some of
the fancier features. NCon relies on the
[TensorOperations](https://github.com/Jutho/TensorOperations.jl) package for
implementation of pair-wise tensor contractions.

## Installation
`Pkg.clone("git://github.com/mhauru/NCon.jl")`

## Usage
```
ncon(L, v; forder=nothing, order=nothing, check_indices=false)
```
The first argument `L` is a Tuple of tensors (multidimensional Arrays).
The second argument `v` is a Tuple of Vectors, one for each tensor in
`L`.
Each `v[i]` consists of Ints, each of which labels an index of `L[i]`.
Positive labels mark indices which are to be contracted (summed over).
So if for instance `v[m][i] == 2` and `v[n][j] == 2`, then the `i`th index of
`L[m]` and the `j`th index of `L[n]` are to be summed over.
Negative labels mark indices which are to remain free (uncontracted).

The keyword argument `order` is an Array of all the positive labels, which
specifies the order in which the pair-wise tensor contractions are to be done.
By default it is `sort(all-positive-numbers-in-v)`. Note that whenever an index
joining two tensors is about to be contracted together, ncon contracts at the
same time all indices connecting these two tensors, even if some of them only
come up later in order.

Correspondingly `forder` specifies the order to which the remaining free
indices are to be permuted.  By default it is `sort(all-negative-numbers-in-v,
rev=true)`, meaning for instance `[-1,-2,...]`.

If `check_indices=true` (by default it's `false`) then checks are performed to
make sure the contraction is well-defined. If not, an `ArgumentError` with a
helpful description of what went wrong is provided.

#### Examples

A matrix product:
```julia
julia> using NCon
julia> A = rand(3,4);
julia> B = rand(4,5);
julia> C = ncon((A, B), ([-1,1], [1,-2]));
julia> size(C)
(3,5)
```
Here the last index of `A` and the first index of `B` are contracted.
The result is a tensor with two free indices, labeled by `-1` and `-2`.
The one labeled with `-1` becomes the first index of the result. If we gave the
additional argument `forder=[-2,-1]` the tranpose would be returned instead.

A more complicated example:
```julia
julia> A = rand(3,4,5);
julia> B = rand(5,3,6,7,6);
julia> C = rand(7,2);
julia> D = ncon((A, B, C), ([3,-2,2], [2,3,1,4,1], [4,-1]));
julia> size(D)
(2,4)
```
By default, the contractions are done in the order [1,2,3,4]. This may not be
the optimal choice, in which case we should specify a better contraction order
as a keyword argument.

Disconnected networks are also possible:
```julia
julia> A = rand(2,3);
julia> B = rand(4);
julia> C = ncon((A, B), ([-3,-2], [-1]));
julia> size(C)
(4,3,2)
```
This is the same as the tensor product of `A` and `B`, with the indices
permuted to the desired order. When contracting disconnected networks, the
connected parts are always contracted first, and their tensor product is taken
at the end.

`L` and `v` may also be a single tensor and its index list, if a trace is taken:
```julia
julia> A = rand(3,2,3);
julia> B = ncon(A, [1,-1,1]);
julia> size(B)
(2,)
```

