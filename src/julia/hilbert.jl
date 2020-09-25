import Base:*
import Base:!
import LinearAlgebra

const TT = ComplexF64
const Z = [TT(1) 0; 0 -1]
const X = [TT(0) 1; 1 0]
const Y = [TT(0) 1im; -1im 0]

# types
const Ket = Array{TT, n} where n
const Bra = Array{TT, n} where n
const Gate = Array{TT, n} where n 
 
struct Repeat
    n::Int
end
function !(n::Int) ::Repeat
    Repeat(n)
end

function ⊗(a::Array{TT, n}, b::Array{TT, m}) ::Array{TT, n + m} where {m, n}
    [aa * bb for aa in a, bb in b]
end
function ⊗(a::Array{TT, n}, b::Repeat) ::Array{TT, b.n * n} where {n}
    if (b.n == 1)
        a
    else
        (a ⊗! (b.n ÷ 2 + b.n % 2)) ⊗ (a ⊗! (b.n ÷ 2))
    end
end

struct Local{m}
    op ::Array{TT, m}
    mapping ::Vector{Int}
end
function Local(op ::Local{m}, mapping)
    om = copy(op.mapping)
    permute!(om, mapping)
    new(op.op, op.mapping) 
end 

function *(a ::Local{m}, b ::Array{TT, n}) ::Array{TT, n} where {n, m}
    perm = [[x for x in 1:ndims(b) if !(((x + 1) ÷ 2) in a.mapping)]..., [2 * x + r for x in a.mapping for r in [-1, 0]]...]
    inv = [1:ndims(b)...]
    invpermute!(inv, perm)
    t = permutedims(b, perm)
    res = a.op * t
    permutedims(res, inv)     
end

function *(a ::Array{TT, n}, b ::Local{m}) ::Array{TT, n} where {n, m}
    perm = [[x for x in 1:ndims(a) if !(((x + 1) ÷ 2) in b.mapping)]..., [2 * x + r for x in b.mapping for r in [-1, 0]]...]
    inv = [1:ndims(a)...]
    invpermute!(inv, perm)
    t = permutedims(a, perm)
    res = t * b.op
    permutedims(res, inv)     
end

function *(a::Array{TT, n}, b::Array{TT, m}) ::Array{TT, max(n,m)} where {m, n}
    @assert size(a, n) == size(b, m-1)
    adim = size(a, n-1)
    bdim = size(a, n)
    cdim = size(b, m)
    arind = repeat([:,], n - 2)
    brind = repeat([:,], m - 2)
    r = [sum(a[arind..., i, j] * b[brind..., j, k] for j in 1:bdim) for i in 1:adim for k in 1:cdim]
    if (ndims(r[1]) > 0)
        reshape(cat([vec(z) for z in r]..., dims=1), (size(r[1])..., adim, cdim))
    else
        reshape(r, (adim, cdim))
    end
end

function *(a::Array{TT, n}, b::Repeat) ::Array{TT, n} where {n}
    if (b.n == 1)
        a
    else
        (a *! (b.n ÷ 2 + b.n % 2)) * (a *! (b.n ÷ 2))
    end
end

function on(op::Array{TT, n}, site::Int, from::Int) ::Array{TT, 2 * from} where {n}
    if (site - 1 > 0)
        lop = I(site - 1) ⊗ op
    else
        lop = op
    end
    if (from - site - n  ÷ 2  + 1 > 0)
        lop ⊗ I(from - site - n  ÷ 2 + 1)
    else
        lop
    end
end


function I(dim::Int, n=2::Int) ::Array{TT, 2 * dim}
    LinearAlgebra.Matrix{TT}(LinearAlgebra.I, n, n) ⊗! dim
end

function norm(a ::Array{TT, n}) where {n}
    LinearAlgebra.norm(a)
end

function tr(ss::Array{Int, 1}, a::Array{TT, n}) ::Array{TT, n - 2 * length(ss)} where {n}
    res = a
    for s in sort(ss, rev=true)
        res = tr(s, res)
    end
    res
end

function tr(s::Int, a::Array{TT, n}) :: Array{TT, n - 2} where {n} 
    @assert size(a, 2 * s) == size(a, 2 * s - 1)
    hdim = size(a, 2 * s)
    sum(a[repeat([:,], 2 * s - 2)..., i, i, repeat([:,], n - 2 * s)...] for i in 1:hdim)
end

function ket(i::Int, n=2::Int) :: Array{TT, 2}
    v = map(j -> j == i ? TT(1) : TT(0), 0:(n-1))
    reshape(v, length(v), 1)
end

function ket(i::Array{Int, 1}, n=2::Int) ::Array{TT, 2 * length(i)}
    if length(i) == 1
        return ket(i[1], n)
    else
        ket(i[1:(n ÷ 2)], n) ⊗ ket(i[(n ÷ 2 + 1):end], n)
    end
end

function bra(i::Int, n=2::Int)
    cj(ket(i, n))
end

function bra(i::Array{Int, 1}, n=2::Int)
    cj(ket(i, n))
end

function cj(a::Array{TT, n}) where {n}
    permutedims(conj.(a), [(x % 2) == 0 ? x - 1 : x + 1 for x in 1:ndims(a)])
end

function to_mat(a::Array{TT, n}) ::Matrix{TT} where {n}
    perm = [[2 * i - 1 for i in 1:(n ÷ 2)]... [2 * i for i in 1:(n ÷ 2)]...]
    rr = permutedims(a, perm)
    reshape(rr, (prod(size(rr)[1:(n ÷ 2)]), prod(size(rr)[((n ÷ 2) + 1):n])))
end

function from_mat(m::Matrix{TT}, space::NTuple{n, Int}) ::Array{TT, 2 * n}  where {n}
    to_space = size(m, 1) == 1 ? ones(Int, n) : space
    from_space = size(m, 2) == 1 ? ones(Int, n) : space
    r = reshape(m, (to_space..., from_space...))
    perm = [i % 2 == 0 ? n + i ÷ 2 : i ÷ 2 + 1 for i in 1:(2 * n)]
    permutedims(r, perm)
end

function dim(a::Array{TT, n}) ::NTuple{n ÷ 2, Int} where {n}
    (maximum(reshape([size(a)...], 2, ndims(a) ÷ 2), dims=1)..., )
end

function hilbert_test()
    I4 = I(4)
    k = ket([1, 1, 1, 1])
    b = bra([1, 1, 1, 1])
    println(b * k) # [1]
    println(to_mat(b) * to_mat(k)) # 1
    println(b * on(Z, 2, 4) * k) # [-1]
    println(from_mat(to_mat(b), dim(b)) - b) # 0
    
    println("Tensor product")
    println("Local product")
    bn = bra([1, 0, 1, 1])
    println(bn * Local(X, [2]) * k)
    println(b * Local(X, [2]) * k)
end


hilbert_test()
