using NCon
using Base.Test
using TensorOperations

A = rand(3,2,5)
B = rand(5,6)
C = rand(2,2,5,4,5)
D = rand(10,8)
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

# Note that the different calls use tuples/Arrays for the arguments in
# different combinations. They should all be equally valid.
con1 = ncon((A, B), ([-1,-2,1], [1,-3]); check_indices=true)
@tensor con1reco[a,b,c] := A[a,b,i] * B[i,c]
@test isapprox(con1, con1reco)

con2 = ncon((A, B, C), ((-1,1,2), (2,-2), (-4,1,3,-3,3)); check_indices=true)
@tensor con2reco[a,b,c,d] := A[a,k,j] * B[j,b] * C[d,k,i,c,i]
@test isapprox(con2, con2reco)

con3 = ncon((A, C), ((-1,2,1), (-2,2,-3,-4,1)); check_indices=true)
@tensor con3reco[a,b,c,d] := A[a,j,i] * C[b,j,c,d,i]
@test isapprox(con3, con3reco)

@test D == ncon(D, [-1,-2]; check_indices=true)

con5 = ncon((D, D), ([-1,-2], [-3,-4]); check_indices=true)
@tensor con5reco[a,b,c,d] := D[a,b] * D[c,d]
@test isapprox(con5, con5reco)

con6 = ncon(E, [1,2,1,2]; check_indices=true)
@tensor con6reco[] := E[i,j,i,j]
@test isapprox(con6, con6reco)

