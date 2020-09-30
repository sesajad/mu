import Base:*, ==, +, -, /, conj, convert

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

H(dim ::Int) ::H = H(Hid(dim), 0)
dim(h ::H) ::Int = h.ref.dim

@testset "H definition" begin
        @test dim(H(4)) == 4
    end

dual(h ::H) ::H = H(h.ref, h.duality + 1)
# assuming duality > 0
idual(h ::H) ::H = H(h.ref, h.duality - 1)

# conj(h ::H) ::H = H(h.ref, 1 - h.duality)
# assumming h.duality is in {0, 1}

hash(h ::H) ::UInt = objectid(h.ref) + 100000007 * h.duality


function ==(a ::H, b ::H)
    a.duality == b.duality && a.ref == b.ref
end

@testset "hash and equality of H" begin
        h = H(4)
        @test hash(dual(h)) != hash(h)
        @test h != dual(h)
        @test !(h == dual(h))
        @test !(h != h)
        @test h == h
        @test h == dual(idual(h))
    end

# ===================
#  Vectors
# ===================

struct Tensor
    value ::Array{TT}

    dim_sites ::Dict{H, Int}
    dim_index ::Vector{H}
end

function Tensor(value ::Array{T, n}, hs ::Vector{H}) ::Tensor where {T <: Number, n}
    Tensor(convert(Array{TT, n}, value), Dict(v => i for (i, v) in enumerate(hs)), hs)
end

rank(t ::Tensor) ::Int = length(t.dim_index)

@testset "Tensor definition" begin
        h = H(2)
        t = Tensor([1, 0], [h])
        @test rank(t) == 1
    end

function *(a ::Tensor, b ::Tensor) ::Tensor
    common_hs = [(aa, bb) for aa in a.dim_index
        for bb in b.dim_index if aa == dual(bb)]

    a_hs:: Vector{H} = [aa for aa in a.dim_index if !(idual(aa) in keys(b.dim_sites))]
    b_hs:: Vector{H} = [bb for bb in b.dim_index if !(dual(bb) in keys(a.dim_sites))]

    a_value = permutedims(a.value,
        [[a.dim_sites[h] for h in a_hs]..., [a.dim_sites[hc] for (hc, _) in common_hs]...])
    b_value = permutedims(b.value,
        [[b.dim_sites[hc] for (_, hc) in common_hs]..., [b.dim_sites[h] for h in b_hs]...])

    a_value = reshape(a_value,
        (prod([dim(h) for h in a_hs]), prod([dim(hc) for (hc, _) in common_hs])))
    b_value = reshape(b_value,
        (prod([dim(hc) for (_, hc) in common_hs]), prod([dim(h) for h in b_hs])))

    c_value = a_value * b_value
    reshape(c_value, ([dim(h) for h in a_hs]..., [dim(h) for h in b_hs]...))

    Tensor(c_value, vcat(a_hs, b_hs))
end

function +(a ::Tensor, b ::Tensor) ::Tensor
    b_value = permutedims(b.value, [b.dim_sites[h] for h in a.dim_index])
    Tensor(a.value + b.value, a.dim_sites, a.dim_index)
end

-(a ::Tensor, b ::Tensor) ::Tensor = a + (-1) * b

dual(t ::Tensor) ::Tensor = Tensor(conj(t.value), [dual(h) for h in t.dim_index])
idual(t ::Tensor) ::Tensor = Tensor(conj(t.value), [idual(h) for h in t.dim_index])

#function conj(t ::Tensor) ::Tensor
#    Tensor(conj(t.value), [conj(h) for h in t.dim_index])
#end

@testset "Tensor basic operations" begin
        h = H(2)
        k = Tensor([1, 0], [h])
        @test rank(k * dual(k)) == 2
        @test rank(dual(k) * k) == 0
    end

*(a ::Tensor, b ::Number) ::Tensor = Tensor(a.value * b, a.dim_sites, a.dim_index)
*(b ::Number, a ::Tensor) ::Tensor = Tensor(a.value * b, a.dim_sites, a.dim_index)
/(a ::Tensor, b ::Number) ::Tensor = Tensor(a.value * b, a.dim_sites, a.dim_index)

pnorm(t ::Tensor, p ::Int=2) ::Number = sum(vec(t.value) .^ p) ^ (1 / p)

convert(::Type{Number}, t ::Tensor) ::Number = t.value[]

@testset "Tensor rank-0/field operations" begin
        h = H(2)
        k0 = Tensor([1, 0], [h])
        k1 = Tensor([0, 1], [h])
        @test pnorm(dual(k0) * k0, 1) == 1
        @test pnorm(dual(k0) * k1, 1) == 0
        @test pnorm(k0 - k0) == 0
    end


# generalizing tr, will lead us to make a tensor,
# a second-rank tensor of non-numeric object, then try to
# do the same operations as for matrices, such as tr, eigen, exp, log ...

# note that it take trace on each (h, dual(h)) pair spaces for h in hs.

function tr(b ::Tensor, hs ::Vector{H}) ::Tensor
    if !isempty(hs)
        h = hs[1]
        h, hc = b.dim_sites[h] < b.dim_sites[dual(h)] ?
            (h, dual(h)) : (dual(h), h)
        pre_i = repeat([:,], b.dim_sites[h] - 1)
        mid_i = repeat([:,], b.dim_sites[hc] - b.dim_sites[h] - 1)
        post_i = repeat([:,], rank(b) - b.dim_sites[hc])
        tr_value = sum([b.value[pre_i..., i, mid_i..., i, post_i...] for i in 1:dim(h)])
        tr_index = [hh for hh in b.dim_index if !(hh in (h, hc))]
        if isempty(tr_index)
            tr_value = fill(tr_value) # fix the 0-dim case
        end
        tr(Tensor(tr_value, tr_index), hs[2:end])
    else
        b
    end
end

@testset "Tensor maps operations" begin
        h = H(2)
        k = Tensor([1, 0], [h])
        @test convert(Number, tr(k * dual(k), [h])) == 1
    end

