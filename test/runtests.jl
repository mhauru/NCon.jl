using NCon
using Test
using TensorOperations

# Some random tensors of various sizes to be used in the tests.
A = rand(3,2,5)
B = rand(5,6)
C = rand(2,2,5,4,5)
D1, D2, D3 = rand(10,8), rand(10,8), rand(10,8)
E = rand(3,2,3,2)
I = rand(-100:100, 3,2)
J = rand(-100:100, 5,2,3,5)

# Test various in valid calls. check_indices should catch all of these.
# More index lists than tensors.
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,1], [1,-3], [-4]); check_indices=true))
# Fewer index lists than tensors.
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,1]); check_indices=true))
# Contraction index appearing thrice.
@test_throws(ArgumentError,
             ncon((A, B), ([-1,1,1], [1,-3]); check_indices=true))
# Free index appearing twice.
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-1,1], [1,-3]); check_indices=true))
# Zero in index lists.
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,0], [0,-3]); check_indices=true))
# Incompatible contraction indices (different dimension).
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,1], [-3,1]); check_indices=true))
# Negative number in order.
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); order=[1,2,-5],
                  check_indices=true))
# Not all contraction indices in order.
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); order=[1],
                  check_indices=true))
# An extraneous positive number in order.
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); order=[1,2,3],
                  check_indices=true))
# Positive number in forder.
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); forder=[-1,-2,1],
                  check_indices=true))
# Not all free indices in forder.
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); forder=[-1],
                  check_indices=true))
# An extraneous negative number in forder.
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); forder=[-1,-2,-5],
                  check_indices=true))

# Do various contractions using ncon and the @tensor macro, as well as a pure
# index permutation and a noop, and check that the results match.
# Note that the different calls use tuples/Arrays for the arguments in
# different combinations. They should all be equally valid.
con = ncon((A, B), ([-1,-2,1], [1,-3]); check_indices=true)
@tensor reco[a,b,c] := A[a,b,i] * B[i,c]
@test isapprox(con, reco)

con = ncon((A, B, C), ((-1,1,2), (2,-2), (-4,1,3,-3,3)); check_indices=true)
@tensor reco[a,b,c,d] := A[a,k,j] * B[j,b] * C[d,k,i,c,i]
@test isapprox(con, reco)

con = ncon((A, C), ((-1,2,1), (-2,2,-3,-4,1)); check_indices=true)
@tensor reco[a,b,c,d] := A[a,j,i] * C[b,j,c,d,i]
@test isapprox(con, reco)

@test permutedims(C, [3,1,2,4,5]) == ncon(C, [-11,-22,-33,-44,-55];
                                          forder=[-33,-11,-22,-44,-55],
                                          check_indices=true)

@test D1 == ncon(D1, [-1,-2]; check_indices=true)

con = ncon((D1, D2), ([-1,-2], [-3,-4]); check_indices=true)
@tensor reco[a,b,c,d] := D1[a,b] * D2[c,d]
@test isapprox(con, reco)

con = ncon((D1, D2, D3), ([-1,-2], [-6,-5], [-3,-4]);
           forder=[-2,-4,-5,-1,-6,-3], check_indices=true)
@tensor reco[-2,-4,-5,-1,-6,-3] := D1[-1,-2] * D2[-6,-5] * D3[-3,-4]
@test isapprox(con, reco)

con = ncon(E, [1,2,1,2]; check_indices=true)
@tensor reco[] := E[i,j,i,j]
@test isapprox(con, reco)

con = ncon((I, J), ([1,2], [-2,2,1,-1]); check_indices=true)
reco = tensorcontract(I, [:i,:j], J, [:b,:j,:i,:a], [:a,:b])
@test isapprox(con, reco)

con = ncon(J, [1,-1,-2,1]; check_indices=true)
@tensor reco[a,b] := J[i,a,b,i]
@test isapprox(con, reco)

# test whether it works for non-blastypes
A = Float16.(A)
B = Float16.(B)
C = Float16.(C)
con = ncon((A, B, C), ((-1,1,2), (2,-2), (-4,1,3,-3,3)); check_indices=true)
@tensor reco[a,b,c,d] := A[a,k,j] * B[j,b] * C[d,k,i,c,i]
@test isapprox(con, reco)
