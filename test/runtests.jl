using NCon
using Base.Test
using TensorOperations

A = rand(3,2,5)
B = rand(5,6)
C = rand(2,2,5,4,5)
D1, D2, D3 = rand(10,8), rand(10,8), rand(10,8)
E = rand(3,2,3,2)

@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,1], [1,-3], [-4]); check_indices=true))
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,1]); check_indices=true))
@test_throws(ArgumentError,
             ncon((A, B), ([-1,1,1], [1,-3]); check_indices=true))
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-1,1], [1,-3]); check_indices=true))
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,0], [0,-3]); check_indices=true))
@test_throws(ArgumentError,
             ncon((A, B), ([-1,-2,1], [-3,1]); check_indices=true))
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); order=[1,2,-5],
                  check_indices=true))
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); order=[1],
                  check_indices=true))
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); order=[1,2,3],
                  check_indices=true))
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); forder=[-1,-2,1],
                  check_indices=true))
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); forder=[-1],
                  check_indices=true))
@test_throws(ArgumentError,
             ncon((A, E), ([2,1,-3], [-1,-2,2,1]); forder=[-1,-2,-5],
                  check_indices=true))

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

