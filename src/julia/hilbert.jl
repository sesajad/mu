import Base:*, ==, +, -, /, conj, convert, size

import LinearAlgebra:I

using Test

const TT = ComplexF64

# ===================
#    Hilbert Space
# ===================

# this structure is the unqiue identifier of each hilbert space
# note that this is an internal structure.

struct H
    uid ::UInt
    dim ::Int
    duality ::Int
end
H(uid ::UInt, dim ::Int) = H(uid, dim, 0)
H(dim ::Int) ::H = H(rand(UInt), dim, 0)
dim(h ::H) ::Int = h.dim

@testset "H definition" begin
        @test dim(H(4)) == 4
    end

dual(h ::H) ::H = H(h.uid, h.dim, h.duality + 1)

idual(h ::H) ::H = H(h.uid, h.dim, h.duality - 1)

conj(h ::H) ::H = H(h.uid, h.dim, 1 - h.duality)
# assumming h.duality is in {0, 1}

HASHPRIME = 100000007
hash(h ::H) ::UInt = h.uid + HASHPRIME * h.duality


function ==(a ::H, b ::H)
    a.duality == b.duality && a.uid == b.uid
end

@testset "hash and equality of H" begin
        h = H(4)
        @test hash(dual(h)) != hash(h)
        @test h != dual(h)
        @test !(h == dual(h))
        @test !(h != h)
        @test h == h
        @test h == dual(idual(h))
        @test conj(h) == dual(h)
        @test conj(dual(h)) == h
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
    # WARN not works with duplicate H in t
    Tensor(convert(Array{TT, n}, value), Dict(v => i for (i, v) in enumerate(hs)), hs)
end

space(t ::Tensor) ::Vector{H} = t.dim_index

order(t ::Tensor) ::Int = length(t.dim_index)
size(t ::Tensor) ::Tuple = size(t.value)

@testset "Tensor definition" begin
        h = H(2)
        t = Tensor([1, 0], [h])
        @test order(t) == 1
    end

function *(a ::Tensor, b ::Tensor) ::Tensor
    # WARN not works with duplicate H in t
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
    c_value = reshape(c_value, ([dim(h) for h in a_hs]..., [dim(h) for h in b_hs]...))

    Tensor(c_value, vcat(a_hs, b_hs))
end

function +(a ::Tensor, b ::Tensor) ::Tensor
    b_value = permutedims(b.value, [b.dim_sites[h] for h in a.dim_index])
    Tensor(a.value + b.value, a.dim_sites, a.dim_index)
end

-(a ::Tensor, b ::Tensor) ::Tensor = a + (-1) * b

dual(t ::Tensor) ::Tensor = Tensor(conj(t.value), [dual(h) for h in t.dim_index])
idual(t ::Tensor) ::Tensor = Tensor(conj(t.value), [idual(h) for h in t.dim_index])

function conj(t ::Tensor) ::Tensor
    Tensor(conj(t.value), [conj(h) for h in t.dim_index])
end

@testset "Tensor basic operations" begin
        h = H(2)
        k = Tensor([1, 0], [h])
        @test order(k * dual(k)) == 2
        @test order(dual(k) * k) == 0
    end

*(a ::Tensor, b ::Number) ::Tensor = Tensor(a.value * b, a.dim_sites, a.dim_index)
*(b ::Number, a ::Tensor) ::Tensor = Tensor(a.value * b, a.dim_sites, a.dim_index)
/(a ::Tensor, b ::Number) ::Tensor = Tensor(a.value / b, a.dim_sites, a.dim_index)
==(a ::Tensor, b ::Tensor) ::Bool = pnorm(a - b, 1) == 0
pnorm(t ::Tensor, p ::Int=2) ::Number = sum(vec(t.value) .^ p) ^ (1 / p)

convert(::Type{Number}, t ::Tensor) ::Number = t.value[]

@testset "Tensor order-0/field operations" begin
        h = H(2)
        k0 = Tensor([1, 0], [h])
        k1 = Tensor([0, 1], [h])
        @test pnorm(dual(k0) * k0, 1) == 1
        @test convert(Number, dual(k0) * k0) == 1
        @test pnorm(dual(k0) * k1, 1) == 0
        @test pnorm(k0 - k0) == 0
        @test k0 == k1
    end

function morph_tensor(from ::Vector{H}, to ::Vector{H}) ::Tensor
    hs = [[dual(h) for h in from]..., to...]
    @assert prod(dim, from) == prod(dim, to)
    d = prod([dim(h) for h in from])
    idn = reshape(Matrix(I, d, d), Tuple(dim(h) for h in hs))
    Tensor(idn, hs)
end

function morph(t ::Tensor, from ::Vector{H}, to ::Vector{H}) ::Tensor
    # WARN not works with duplicate H in t
    rem:: Vector{H} = [hh for hh in t.dim_index if !(hh in from)]
    t_value = permutedims(t.value,
        [[t.dim_sites[h] for h in rem]..., [t.dim_sites[f] for f in from]...])
    t_value = reshape(t_value, ([dim(h) for h in rem]..., [dim(h) for h in to]...))
    Tensor(t_value, [rem..., to...])
end

@testset "Hilbert space morphing" begin
        h1 = H(4)
        h2a = H(2)
        h2b = H(2)
        t = Tensor([1,0,1,0], [h1])
        a1 = Tensor([1, 0], [dual(h2a)])
        a2 = Tensor([1, 1], [h2b])

        t2p = morph_tensor([h1], [h2a, h2b]) * t
        @test (a1 * t2p) == a2

        t2m = morph(t, [h1], [h2a, h2b])
        @test (a1 * t2m) == a2
    end

# a more general and simple way to implement tr
# other operations as for matrices, such as tr, eigen, exp, log ...

# note that it take trace on each (h, dual(h)) pair spaces for h in hs.

function tr_tensor(hs ::Vector{H}) ::Tensor
    prod([Tensor(Matrix(I, dim(h), dim(h)), [dual(dual(h)), dual(h)]) for h in hs])
end

function tr(b ::Tensor, hs ::Vector{H}) ::Tensor
    if !isempty(hs)
        h = hs[1]
        h, hc = b.dim_sites[h] < b.dim_sites[dual(h)] ?
            (h, dual(h)) : (dual(h), h)
        pre_i = repeat([:,], b.dim_sites[h] - 1)
        mid_i = repeat([:,], b.dim_sites[hc] - b.dim_sites[h] - 1)
        post_i = repeat([:,], order(b) - b.dim_sites[hc])
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
        k0 = Tensor([1, 0], [h])
        k1 = Tensor([0, 1], [h])
        op = k0 * dual(k0) + 0.5 * k1 * dual(k1)
        @test convert(Number, tr(op, [h])) == 1.5
        @test convert(Number, tr_tensor([h]) * op) == 1.5
    end

