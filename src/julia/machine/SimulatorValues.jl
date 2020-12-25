module SimulatorValues

using AbstractMachine
using FiniteSpace
import FiniteSpace:DenseVector
using Types
import Types:decompose
using Typed

import Base:repeat, /, convert, +, *
using Test


# ===================
#     PureStates
# ===================

struct Val{T} <: PureState{T}
    t ::DenseVector
end


# defintions
val(t ::Type{T}, f ::Function) where {T} = # f : T -> Complex
    Val{T}(DenseVector([f(tt) for tt in iter(T)], convert(Vector{H}, flatten(global_hilbert(T))))) ::Val{T}
# WARNING fock and schrodingers are unsupported

# hard definitions
val(t ::Set{T}) where {T} = val(T, tt -> ((tt in t) ? (1 / sqrt(length(t))) : 0)) ::Val{T}
val(t ::T)  where {T} = val(Set{T}([t])) ::Val{T}

convert(::Type{Val{T}}, t ::T) where {T} = val(t) ::Val{T}
convert(::Type{Val{T}}, t ::Set{T}) where {T} = val(t) ::Val{T}

#actions
function cast(::Type{V}, v ::Val{T}) ::Val{V} where {T, V}
    Val{V}(morph(v.t, flatten(global_hilbert(T)), flatten(global_hilbert(V))))
end

function compose(::Type{T}, vs ::Val...) ::Val{T} where {T}
    froms = [flatten(global_hilbert(t)) for t in decompose(T)]
    tos = [flatten(h) for h in global_hilbert(T)]
    Val{T}(prod([morph(v.t, f, t) for (v, f, t) in zip(vs, froms, tos)]))
end

decompose(v ::Val{T}) where {T} = error("mixed are not implemented")

# hard actions
+(a ::Val{T}, b ::Val{T}) where {T} = Val{T}(a.t + b.t) ::Val{T}
-(a ::Val{T}, b ::Val{T}) where {T} = (a + (-1) * b) ::Val{T}
*(a ::Val{T}, b ::Number) where {T} = Val{T}(a.t * b) ::Val{T}
*(b ::Number, a ::Val{T}) where {T} = Val{T}(a.t * b) ::Val{T}
/(a ::Val{T}, b ::Number) where {T} = Val{T}(a.t / b) ::Val{T}
normalize(v ::Val{T}) where {T} = (v / pnorm(v.t)) :: Val{T}

@testset "values (aka rvalues)" begin
    v0 = val(Bool, x -> 1/sqrt(2))
    v1 = val(false)
    v2 = val(Set([false, true]))
    v3 = val(true)
    @test typeof(v1) == typeof(v2) == typeof(v3) == Val{Bool}
    @test v0.t == v2.t
    @test normalize(v1 + v3).t == v2.t
    @test convert(Number, conj(v1.t) * v2.t) == 1 / sqrt(2)
    v4 = val((true, false))
    @test compose(Tuple{Bool, Bool}, v3, v1).t == v4.t
end


# ===================
#        Maps
# ===================

struct Map{T, V} <: AbsMap{T, V}
    t ::DenseVector
end

function (m ::Map)(from :: Vector{H}, to ::Vector{H}) ::DenseVector
    t = morph(m.t, flatten(conj(global_hilbert(T))), conj(frm))
    t = morph(t, flatten(global_hilbert(V)), to)
    # for th sake of identity, ignore errors
end

# definitions
function identity(::Type{T}) ::Map{T, T} where {T}
    Map{T, T}(1)
end

function equals(::Type{T}) ::Map{Tuple{T, T}, Bool} where {T}
    map(Tuple{T, T} => Bool, (x, y) -> x == y)
end

function successor(::Type{T}) ::Map{T, T} where {T <: Number}
    map(T => T, x -> x + 1)
end # warn

function fourier(::Type{T}) ::Map{T, T} where {T <: Number}
    f = x::T -> val(T, y::T -> exp(2im * pi * x * y / prod(size(T)))  / sqrt(prod(size(T))) )
    map(T => T, f)
end

function repeat(::Type{T}, n ::Int) ::Map{T, NTuple{m, T}} where {m, T}
    map(T => NTuple{n, T}, x -> ntuple(_ -> x, n))
end

# hard definitions
function map(p ::Pair, f ::Function) # f: T -> convertible to Val{V}
    T, V = p
    DenseVector = sum([convert(Val{T}, f(tt)).t * conj(val(tt).t) for tt in iter(T)])
    Map{T, V}(DenseVector)
end
# WARNING fock and schrodingers are unsupported

# actions

# if m is bijective
function inv(m ::Map{T, V}) ::Map{V, T} where {T, V}
    Map{V, T}(conj(m.t))
end


# TODO, an idea!
Try{T} = Union{T, Exception}
function pseudoinv(m ::Map{T, V}) ::Map{V, Try{T}} where {T, V}
    error("not implemented")
end

# TODO!
function convert(::Type{Map{Unit, V}}, Val{V}) = error("not implemented")

function cast(p ::Pair, m ::Map{U, W}) ::Map where {U, W}
    T, V = p
    Map{T, V}(m(flatten(global_hilbert(T)), flatten(global_hilbert(V))))
end

function compose(p ::Pair, maps ::Map...)
    T, V = p
    tfroms = [conj(flatten(global_hilbert(t))) for t in decompose(T)]
    ttos = [conj(flatten(h)) for h in global_hilbert(T)]
    maps = [v(f, t) for (v, f, t) in zip(maps, tfroms, ttos)]

    vfroms = [flatten(global_hilbert(t)) for t in decompose(T)]
    vtos = [flatten(h) for h in global_hilbert(T)]
    maps = [v(f, t) for (v, f, t) in zip(maps, vfroms, vtos)]

    Map{T, V}(prod(maps))
end


decompose(m ::Map) = error("channels are not implemented")

function *(a ::Map{T, V}, b ::Map{V, U}) ::Map{T, U} where {T, V, U}
    Map{T, U}(a.t * b.t)
end

# hard actions
function switch(expr ::Map{T, S}, pairs::Pair{S, Map{U, V}}...) ::Map{Tuple{T, V}, Tuple{T, U}} where {T, S, U, V}
    hta, hv = global_hilbert(Tuple{T, V})
    htb, hu = global_hilbert(Tuple{T, U})
    ht = global_hilbert(T)
    hs = global_hilbert(S)

    tt = repeat(T, 2)(flatten(hta), flatten([htb, ht]))
    tt *= cond(ht, hs)
    tt *= sum([conj(val(v).t) * c(hv, hu) for (v, c) in pairs])
    Map{Tuple{T, V}, Tuple{T, U}}(tt)
end

function ifte(cond ::Map{T, Bool}, thenm ::Map{U, V}, elsem ::Map{U, V}) ::Map{Tuple{T, V}, Tuple{T, U}} where {T, U, V}
    switch(cond, true => thenm, false => elsem)
end

#for subspace mapping
# has an additional reconstruction cond
#function iffi(cond ::Map{T, Bool}, thenm ::Map{Tuple{T, U}, Tuple{T, V}}, elsem ::Map{Tuple{T, U}, Tuple(T, V}}, recond ::Map{Tuple{T, U}, Bool}) ::Map{Tuple{T, V}, Tuple{T, U}} where {T, U, V}

# alternative proposal
#function iffi(cond ::Map{T, Bool}, thenm ::Map{T, V}, else ::Map{T, V}, recond ::Map{V, B})

@testset "maps (aka lambdas)" begin
    f = val(false)
    t = val(true)
    hadamard = map(Bool => Bool, x -> (val(false) + (x ? -1 : 1) * val(true)) / sqrt(2))
    @test convert(Number, conj(t.t) * hadamard.t * f.t) == 1 / sqrt(2)
    @test convert(Number, conj(f.t) * hadamard.t * f.t) == 1 / sqrt(2)
    @test hadamard.t ≈ fourier(Bool).t
    @test (identity(Bool) * identity(Bool)).t == identity(Bool).t
end



# ===================
#      Measures
# ===================



struct Op{T} # positive operator on T
    t ::DenseVector
end
op(v ::T) where {T} = op(convert(Val{T}, v))
op(v ::Val{T}) where{T} = Op{T}(v.t * conj(v.t))
op(v ::Set{T}) where {T} = sum([op(vv) for vv in v])
op(m ::Map{Val{T}, Real}) where {T} = Op{T}(sum([v.t * r * conj(v.t) for (v, r) in m]...))

+(a ::Op{T}, b ::Op{T}) where {T} = Op{T}(a.t + b.t) ::Op{T}
-(a ::Op{T}, b ::Op{T}) where {T} = Op{T}(a.t - b.t) ::Op{T}
*(a ::Op{T}, b ::Number) where {T} = Op{T}(a.t * b) ::Op{T}
*(a ::Number, b ::Op{T}) where {T} = Op{T}(a * b.t) ::Op{T}

one(::Type{T}) where {T} = Op{T}(1)

function (m ::Op)(on :: Vector{H}) ::DenseVector
    t = morph(m.t, flatten(conj(global_hilbert(T))), conj(on))
    t = morph(t, flatten(global_hilbert(T)), on)
end

struct Measure{T, F}
    d ::Dict{Op{T}, F}
end

measurement(d ::Dict{Op{T}, F}) where {T, F} = Measure{T, F}(d)
default_measurement(::Type{T}) where {T} = Dict(i => op[i] for i in iter(T))

export val, Val, map, Map, fourier

end
