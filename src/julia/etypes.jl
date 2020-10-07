# enumeratable (linearly) and known-sized
# decompose(T), index(::T)::Int, iter(T), size(T), item(::Int)::T
import Base:size

decompose(::Type{Bool}) = Bool
size(::Type{Bool}) = (2,) # size of the iterator
iter(::Type{Bool}) = [false, true]
index(t ::Bool) ::Int = convert(Int, t) + 1
item(i ::Int) ::Bool = iter(Bool)[i]

@testset "Boolean iteratable type" begin
    @test size(Bool) == (2,)
    @test all(x -> item(index(x)) == x, iter(Bool))
    @test index(false) == 1
    @test item(2) == true
end

decompose(::Type{NTuple{n, T}}) where {n, T} = Tuple(T for _ in 1:n)
size(::Type{NTuple{n, T}}) where {n, T} = Tuple(v for v in size(T) for _ in 1:n)
function iter(::Type{NTuple{n, T}}) where {n, T}
    if (n == 1)
        [(v,) for v in iter(T)]
    else
        [(v, rest...) for v in iter(T), rest in iter(NTuple{n - 1, T})]
    end
end
index(t ::NTuple{n, T}) where {n, T} = Tuple(index(v) for v in t)
item(i ::NTuple{n, Int}) where {n, T} = Tuple(item{T}(v) for v in i)


@testset "Composite iteratable type" begin
    @test size(Tuple{Bool, Bool}) == (2, 2)
end
