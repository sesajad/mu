import Base:*, ==, +, -

using Test

const TT = ComplexF64

# ===================
#    Hilbert Space
# ===================

# this structure is the unqiue identifier of each hilbert space
# note that this is an internal structure.
struct Hid
    dim ::Int
end

struct H
    ref ::Hid
    duality ::Int
end

function H(dim ::Int) ::H
    H(Hid(dim), 0)
end

function dim(h ::H) ::Int
    h.ref.dim
end

@testset "H definition" begin
        @test dim(H(4)) == 4
    end

# for a system with conj^2 != I with should use conjl, conjr
function conj(h ::H) ::H
    H(h.ref, 1 - h.duality)
end

# assuming duality is in {0, 1}
function hash(h ::H)
    objectid(h.ref) + 100000007 * h.duality
end

function ==(a ::H, b ::H)
    a.duality == b.duality && a.ref == b.ref
end

@testset "hash and equality of H" begin
        h = H(4)
        @test hash(conj(h)) != hash(h)
        @test h != conj(h)
        @test !(h == conj(h))
        @test !(h != h)
        @test h == h
        @test h == conj(conj(h))
    end

# ===================
#  Vectors (Tensors)
# ===================

struct Tensor
    value ::Array{TT}

    dim_sites ::Dict{H, Int}
    dim_index ::Vector{H}
end

function Tensor(value ::Array{TT, n}, hs ::Vector{H}) ::Tensor where {n}
    Tensor(value, Dict(enumerate(hs)...), hs)
end

function dim(t ::Tensor) ::Int
    length(t.dim_index)
end

@testset "Tensor definition" begin
    end

function *(a ::Tensor, b ::Tensor) ::Tensor
    common_hs = [(aa, bb) for aa in dim_index
        for bb in b.dim_index if aa == conj(bb)]

    a_hs = [aa for aa in a.dim_index if !aa in keys(b.dim_sites)]
    b_hs = [bb for bb in b.dim_index if !bb in keys(a.dim_sites)]

    a_value = permutedims(a.value,
        [[a.dim_sites[h] for h in a_hs]..., [a.dim_sites[hc] for (hc, _) in common_hs]...])
    b_value = permutedims(b.value,
        [[b.dim_sites[hc] for (_, hc) in common_hs]...], [b.dim_sites[h] for h in b_hs]...)

    a_value = reshape(a_value,
        (prod([dim(h) for h in a_hs]), prod([dim(h) for (hc, _) in common_hs])))
    b_value = reshape(b_value,
        (prod([dim(h) for (_, hc) in common_hs]), prod([dim(h) for h in b_hs])))

    c_value = a_value * b_value
    reshape(c_value, [[dim(h) for h in a_hs]..., [dim(h) for h in b_hs]...])

    Tensor(c_value, [a_hs..., b_hs...])
end

function +(a ::Tensor, b ::Tensor) ::Tensor
    b_value = permutedims(b.value, [b.dim_sites[h] for h in a.dim_index])
    Tensor(a.value + b.value, a.dim_sites, a.dim_index)
end

function *(a ::Tensor, b ::TT) ::Tensor
    Tensor(a.value * b, a.dim_sites, a.dim_index)
end

function *(b ::TT, a ::Tensor) ::Tensor
    Tensor(a.value * b, a.dim_sites, a.dim_index)
end

function /(a ::Tensor, b ::TT) ::Tensor
    Tensor(a.value * b, a.dim_sites, a.dim_index)
end

function -(a ::Tensor, b ::Tensor) ::Tensor
    a + (-1) * b
end

function conj(t ::Tensor)
    Tensor(conj(value), [conj(h) for h in t.dim_index])
end

function lnorm(t ::Tensor, p ::Int=2)
    sum(vector(t) .^ p) ^ (1/p)
end

function tr(b ::Tensor, hs ::Vector{H})
    if !empty(hs)
        h = head(hs)
        h = b.dim_states[h] < dim_states[conj(h)] ? h : conj(h)
        pre_i = repeat((:,), b.dim_sites[h] - 1)
        mid_i = repeat((:,), b.dim_sites[conj(h)] - b.dim_sites[h] - 1)
        post_i = repeat((:,), dim(b) - b.dim_sites[conj(h)])
        tr_value = sum([b[pre_i..., i, mid_i..., i, post_i...] for i in dim(h)])
        tr_index = [hh for hh in b.index if hh != h && hh != conj(h)]
        tr(Tensor(tr_value, tr_index), tail(hs))
    else
        b
    end
end

# Todo, cast from, and more than that to TT

