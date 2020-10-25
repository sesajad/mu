include("abstract.jl")

import LinearAlgebra:I

using Test


# ===================
#    Hilbert Space
# ===================

# this structure is the unqiue identifier of each hilbert space
# note that this is an internal structure.

struct H <: AbsHilbert
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

HASHPRIME = 100000007
hash(h ::H) ::UInt = h.uid + HASHPRIME * h.duality
function ==(a ::H, b ::H)
    a.duality == b.duality && a.uid == b.uid
end


dual(h ::H) ::H = H(h.uid, h.dim, h.duality + 1)
idual(h ::H) ::H = H(h.uid, h.dim, h.duality - 1)
conj(h ::H) ::H = H(h.uid, h.dim, 1 - h.duality)
# assumming h.duality is in {0, 1}

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

struct DenseVector <: AbsVector
    value ::Array{TT}

    dim_sites ::Dict{H, Int}
    dim_index ::Vector{H}
end

function DenseVector(value ::Array{T, n}, hs ::Vector{H}) ::DenseVector where {T <: Number, n}
    # WARN not works with duplicate H in t
    DenseVector(convert(Array{TT, n}, value), Dict(v => i for (i, v) in enumerate(hs)), hs)
end

space(t ::DenseVector) ::Vector{H} = t.dim_index

order(t ::DenseVector) ::Int = length(t.dim_index)
size(t ::DenseVector) ::Tuple = size(t.value)

@testset "DenseVector definition" begin
    h = H(2)
    t = DenseVector([1, 0], [h])
    @test order(t) == 1
end

==(a ::DenseVector, b ::DenseVector) ::Bool = pnorm(a - b, 1) == 0

function *(a ::DenseVector, b ::DenseVector) ::DenseVector
    # WARN not works with duplicate H in t
    if (order(a) == 0)
        DenseVector(b.value[] * a.value, b.dim_sites, b.dim_index)
    elseif (order(b) == 0)
        DenseVector(a.value * b.value[], a.dim_sites, a.dim_index)
    else
        common_hs = Tuple{H, H}[(aa, bb) for aa in a.dim_index
            for bb in b.dim_index if aa == dual(bb)]

        a_hs = H[aa for aa in a.dim_index if !(idual(aa) in keys(b.dim_sites))]
        b_hs = H[bb for bb in b.dim_index if !(dual(bb) in keys(a.dim_sites))]

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

        DenseVector(c_value, vcat(a_hs, b_hs))
    end
end

function +(a ::DenseVector, b ::DenseVector) ::DenseVector
    b_value = permutedims(b.value, [b.dim_sites[h] for h in a.dim_index])
    DenseVector(a.value + b.value, a.dim_sites, a.dim_index)
end

dual(t ::DenseVector) ::DenseVector = DenseVector(conj(t.value), [dual(h) for h in t.dim_index])
idual(t ::DenseVector) ::DenseVector = DenseVector(conj(t.value), [idual(h) for h in t.dim_index])
conj(t ::DenseVector) ::DenseVector = DenseVector(conj(t.value), [conj(h) for h in t.dim_index])

@testset "DenseVector basic operations" begin
    h = H(2)
    k = DenseVector([1, 0], [h])
    @test order(k * dual(k)) == 2
    @test order(dual(k) * k) == 0
end

pnorm(t ::DenseVector, p ::Int=2) ::Number = sum(vec(t.value) .^ p) ^ (1 / p)

convert(::Type{Number}, t ::DenseVector) ::Number = t.value[]
convert(::Type{DenseVector}, v ::Number) ::DenseVector = DenseVector(fill(v), H[])

@testset "DenseVector order-0/field operations" begin
    h = H(2)
    k0 = DenseVector([1, 0], [h])
    k1 = DenseVector([0, 1], [h])
    @test pnorm(dual(k0) * k0, 1) == 1
    @test convert(Number, dual(k0) * k0) == 1
    @test pnorm(dual(k0) * k1, 1) == 0
    @test pnorm(k0 - k0) == 0
    @test k0 == k1
end

function morph_DenseVector(from ::Vector{H}, to ::Vector{H}) ::DenseVector
    hs = [[dual(h) for h in from]..., to...]
    @assert prod(dim, from) == prod(dim, to)
    d = prod([dim(h) for h in from])
    idn = reshape(Matrix(I, d, d), Tuple(dim(h) for h in hs))
    DenseVector(idn, hs)
end

function morph(t ::DenseVector, from ::Vector{H}, to ::Vector{H}) ::DenseVector
    # WARN not works with duplicate H in t
    rem:: Vector{H} = [hh for hh in t.dim_index if !(hh in from)]
    t_value = permutedims(t.value,
        [[t.dim_sites[h] for h in rem]..., [t.dim_sites[f] for f in from]...])
    t_value = reshape(t_value, ([dim(h) for h in rem]..., [dim(h) for h in to]...))
    DenseVector(t_value, [rem..., to...])
end

@testset "Hilbert space morphing" begin
    h1 = H(4)
    h2a = H(2)
    h2b = H(2)
    t = DenseVector([1,0,1,0], [h1])
    a1 = DenseVector([1, 0], [dual(h2a)])
    a2 = DenseVector([1, 1], [h2b])

    t2p = morph_DenseVector([h1], [h2a, h2b]) * t
    @test (a1 * t2p) == a2

    t2m = morph(t, [h1], [h2a, h2b])
    @test (a1 * t2m) == a2
end

# a more general and simple way to implement tr
# other operations as for matrices, such as tr, eigen, exp, log ...

# note that it take trace on each (h, dual(h)) pair spaces for h in hs.

function tr_DenseVector(hs ::Vector{H}) ::DenseVector
    prod([DenseVector(Matrix(I, dim(h), dim(h)), [dual(dual(h)), dual(h)]) for h in hs])
end

function tr(b ::DenseVector, hs ::Vector{H}) ::DenseVector
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
        tr(DenseVector(tr_value, tr_index), hs[2:end])
    else
        b
    end
end

@testset "DenseVector maps operations" begin
    h = H(2)
    k0 = DenseVector([1, 0], [h])
    k1 = DenseVector([0, 1], [h])
    op = k0 * dual(k0) + 0.5 * k1 * dual(k1)
    @test convert(Number, tr(op, [h])) == 1.5
    @test convert(Number, tr_DenseVector([h]) * op) == 1.5
end

