"""
A module for the function ncon, which does contractions of several tensors.
"""
module NCon
using TensorOperations
using LinearAlgebra: BlasFloat
export ncon

"""
    ncon(L, v; forder=nothing, order=nothing, check_indices=false)

Takes a tuple of tensors L and a tuple of Vector{Integer}s v that specifies
how these tensors form a network, and contracts this network. The return value
is the tensor that is formed as the result of this contraction.

More specifically:
L = (A1, A2, ..., Ap) is a tuple of tensors or a single tensor.

v = (v1, v2, ..., vp) is a tuple of Arrays of indices, each of which
corresponds to one of the tensors in L. For instance, if v1 = (3, 4, -1) and
v2 = (-2, -3, 4), then the second index of A1 and the last index of A2 are
contracted together, because they are both labeled by 4. All positive numbers
label indices that are to be contracted and all negative numbers label indices
that are to remain open. The open indices will be the remaining indices of the
tensor that is formed. The index lists v1, v2, etc. may also be tuples instead
of Arrays, and v may consist of a single index list v1, if there's only one
tensor in the contraction (a trace).

order, if present, contains a tuple or Array of all positive indices - if not
(1, 2, 3, 4, ...) by default. This is the order in which the contractions are
performed. However, whenever an index joining two tensors is about to be
contracted together, ncon contracts at the same time all indices connecting
these two tensors, even if some of them only come up later in order.

forder, if present, contains the final ordering of the uncontracted indices
- if not, (-1, -2, ...) by default.

If check_indices is true (by default it's false) then checks are performed to
make sure the contraction is well-defined. If not, an ArgumentError with a
helpful description of what went wrong is provided.
"""

function ncon(L, v; forder=nothing, order=nothing, check_indices=true)
    # We want to handle the tensors as an Vector{AbstractArray} and v as a
    # Vector{Vector{Number}. If either one is a single AbstractArray or
    # Vector{Integer} then wrap them in a Vector, othewise collect to make sure
    # tuples and other iterables are turned into Vectors.
    L = (isa(L, AbstractArray{T} where T <: Number) ?
         AbstractArray[L] : AbstractArray[L...])
    v = (isa(v, Vector{T} where T <: Number) ?
         [collect(v)] : collect(map(collect, v)))

    order === nothing && (order = create_order(v))
    forder === nothing && (forder = create_forder(v))
    check_indices && do_check_indices(L, v, order, forder)

    # Keep looping until there is only one tensor left. At each iteration
    # either two tensors will be contracted together, or a trace will be taken
    # over the indices of some tensor.
    while length(order) > 0
        tcon = get_tcon(v, order[1])  # tcon = tensors to be contracted

        if length(tcon) == 1  # This contraction is a trace.
            t = tcon[1]
            newA = trace(L[t], v[t])
        else
            t1, t2 = tcon
            newA = con(L[t1], v[t1], L[t2], v[t2])
        end
        # Find the indices icon that were contracted.
        icon = get_icon(v, tcon)
        # Find the indices of the new tensor.
        newv = find_newv(v, tcon, icon)
        push!(L, newA)
        push!(v, newv)
        for i in sort(tcon, rev=true)
            # Delete the contracted tensors and indices from the lists.
            # tcon is reverse sorted so that tensors are removed starting from
            # the end of L, otherwise the order would get messed.
            deleteat!(L, i)
            deleteat!(v, i)
        end
        renew_order!(order, icon)  # Update order
    end
    Alast = multiply_final(L, v, forder)
    return Alast
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


""" Identify all unique, positive indices and return them sorted. """
create_order(v) = sort(filter!(x -> x > 0, unique(reduce(vcat, v))))


"""
Identify all unique, negative indices and return them reverse sorted (-1 first).
"""
create_forder(v) = sort(filter!(x -> x < 0, unique(reduce(vcat, v))), rev=true)


"""
Return a Vector of indices for L of the tensors that have n as their leg.
"""
get_tcon(v, n) = findall(x -> in(n, x), v)

"""
Return a Vector of indices that are to be contracted when contractions between
the two tensors numbered in tcon are contracted.
"""
function get_icon(v, tcon)
    if length(tcon) == 1
        # This is a trace.
        inds = v[tcon[1]]
        # Find all the elements that occur twice in inds.
        return filter!(x -> count(==(x), inds) > 1, unique(inds))
    else
        return unique!(reduce(intersect, v[tcon]))
    end
end


"""
Find the list of indices for the new tensor after contraction of indices icon
of the tensors tcon.
"""
find_newv(v, tcon, icon) = filter!(!in(icon), reduce(vcat, v[tcon]))


""" Update order by removing the contracted indices from it. """
renew_order!(order, icon) = filter!(!in(icon), order)


"""
Check that
1) the number of tensors in L matches the number of index lists in v.
2) every tensor is given the right number of indices.
3) order only has positive numbers in it and forder only negative.
4) every contracted index is featured exactly twice and every free index
   exactly once.
5) That every index appears exactly once in either order or forder.
6) the dimensions of the two ends of each contracted index match.
7) 0 is not in v.
"""
function do_check_indices(L, v, order, forder)
    #1)
    if length(L) != length(v)
        msg = "In ncon, the number of tensors ($(length(L)))"*
              " does not match the number of index lists ($(length(v)))."
        throw(ArgumentError(msg))
    end

    #2)
    for (i, (A, inds)) in enumerate(zip(L, v))
        dimcount = ndims(A)
        numinds = length(inds)
        if numinds != dimcount
            msg = "In ncon, length(v[$i])=$(numinds)"*
                  " does not match the numbers of indices of L[$i] ="*
                  " $(dimcount)"
            throw(ArgumentError(msg))
        end
    end

    #3)
    if any(x -> x <= 0, order)
        msg = "In ncon, not all elements in order are positive."
        throw(ArgumentError(msg))
    end
    if any(x -> x >= 0, forder)
        msg = "In ncon, not all elements in forder are negative."
        throw(ArgumentError(msg))
    end

    #4-7)
    v_sum = reduce(vcat, v)
    if 0 in v_sum
        throw(ArgumentError("Zero is not a valid index for ncon"))
    end
    if Set(union(order, forder)) != Set(v_sum)
        msg = "In ncon, the indices in forder and order are not the same as"*
              " the indices in v."*
              "\nforder: $forder\norder: $order\nv: $v"
        throw(ArgumentError(msg))
    end
    # v_pairs provides all valid indices for v, in the same order as the
    # elements of v_sum, e.g.
    # "v_pairs = [[(1,1), (1,2), (1,3), ...], [(2,1), (2,2), (2,3), ...], ...]"
    v_pairs = [(i, j) for (i, s) in enumerate(v) for j in 1:length(s)]
    # For t, o in zip(v_pairs, v_sum), t is the tuple of the number of
    # the tensor and the index and o is the contraction order of that
    # index. We group these tuples by the contraction order.
    order_groups = [[t for (t, o) in zip(v_pairs, v_sum) if o == n]
                     for n in order]
    forder_groups = [[t for (t, fo) in zip(v_pairs, v_sum) if fo == n]
                     for n in forder]
    for (i, o) in enumerate(order_groups)
        if length(o) != 2
            msg = "In ncon, the contracted index"*
                  " $(order[i]) is not featured exactly twice in v."
            throw(ArgumentError(msg))
        else
            A0, ind0 = o[1]
            A1, ind1 = o[2]
            size_A0_ind0 = size(L[A0], ind0)
            size_A1_ind1 = size(L[A1], ind1)
            compatible = (size_A0_ind0 == size_A1_ind1)
            if !compatible
                msg = "In ncon, for the contraction index"*
                      " $(order[i]), the leg $ind0 of tensor $A0"*
                      " (dimension $size_A0_ind0) and leg $ind1 of tensor $A1"*
                      " (dimension $size_A1_ind1) are not compatible."
                throw(ArgumentError(msg))
            end
        end
    end
    for (i, fo) in enumerate(forder_groups)
        if length(fo) != 1
            msg = "In ncon, the free index"*
                  " $(forder[i]) is not featured exactly once in v."
            throw(ArgumentError(msg))
        end
    end

    # All is well if we made it here.
    return true
end


########################################################################
# The following are simple wrappers around TensorOperations functions, #
# but may be replaced with other stuff later.                          #
########################################################################


"""
If vA contains some element repeated twice, the second of these elements is
replaced by m+1. Each time this occurs, m is incremented by one. The result
is a new Array that has no elements repeated twice, and all the new elements
are larger than m.
"""
function change_duplicates(vA::Vector{T}, m=maximum(abs.(vA))) where {T <: Number}
    s = Set{T}()
    for (i, el) in enumerate(vA)
        if el in s
            vA = copy(vA)
            m += one(T)
            vA[i] = m
        else
            push!(s, el)
        end
    end
    return vA
end


""" Contract two tensors. """
function con(A, vA, B, vB)
    # tensorcontract can't handle a case where vA or vB includes a partial
    # trace (a repeated index) that is to be performed later.
    # Work around this by changing the labels if this occurs.
    m = maximum(abs.(vcat(vA, vB)))
    vA = change_duplicates(vA, m)
    m = maximum(abs.(vcat(vA, vB)))
    vB = change_duplicates(vB, m)
    if A isa BlasFloat && B isa BlasFloat
        enable_blas()
    else
        disable_blas()
    end
    res = tensorcontract(A, vA, B, vB)
    return res
end


""" Trace over some of the indices of a tensor. """
function trace(A, inds)
    res = tensortrace(A, inds)
    return res
end


"""
Return the tensor product (no contractions) of the tensors in L, with the
indices permuted to the order specified in forder.
"""
function multiply_final(L, v, forder)
    if length(L) == 1
        Anew = tensorcopy(L[1], v[1], forder)
    else
        lengths = map(length, L)
        while length(L) > 1
            # Get the two tensors in L with smallest number of elements.
            i = argmin(lengths)
            A, vA = L[i], v[i]
            deleteat!(L, i)
            deleteat!(v, i)
            deleteat!(lengths, i)
            j = argmin(lengths)
            B, vB = L[j], v[j]
            deleteat!(L, j)
            deleteat!(v, j)
            deleteat!(lengths, j)
            # Multiply them together and permute their indices at the same
            # time. vnew is all the indices of vA and vB, in the order that
            # they appear in forder.
            vnew = intersect(forder, vcat(vA, vB))
            Anew = tensorproduct(A, vA, B, vB, vnew)
            push!(L, Anew)
            push!(v, vnew)
            push!(lengths, length(Anew))
        end
    end
    return Anew
end


end
