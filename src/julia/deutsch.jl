# pure model
# abstract type PureState{T} end 	# T -> Complex 		# norm vector

# abstract type Map{T, V} end 	# V -> PureState{T} 	# unitary
# abstract type Evolution{T} <: Map{T, T} end 	# a special case of map


# abstract type Operator{T} end 	# T x T -> Complex	# hermitian
				# PureState{T} -> Real  
				# or Set{(PureState, Real)}
# abstract type Hamiltonian{T} <: Operator{T} end
# abstract type Measure{T, V} end # Set{(Operator, V)}	# operators must be positive

# probabilistic model
# abstract type MixedState{T} end # Set{(PureState, Real)}

# abstract type Channel{T} end 	# MixedState{T} -> MixedState{T}

# abstract type Dist{T} end 	# T -> Real

# polymorphism casts

# cast: T -> PureState{T}
# cast: Set{T} -> PureState{T}
# cast: PureState{T} -> MixedState{T}
# cast: Evolution{T} -> Channel{T}

# storage model
# abstract type Var{T} end

# constructors
# val: T -> Val{T} (tool)
# val: Set{T} -> Val{T} (tool)
# val: Function{T} -> Val{T}
# map: Function{T, PureState{V}} -> Map{T, V}
# var: like var

# functions
# decompose: ?{CompositeType} -> ?...
# compose: NTuple{CompositeType, ?...} -> ?{Composite}
# cast: ?{T} -> ?{U}

# views
# reinterpret: Var{T} x Map{T, V} -> Var{V}

# Modifiers of global state

# _register!: MixedState{T} -> VirtualMixedState{T}
# _entangle!: list of blocks to entangle

# apply!: Var{T} x Channel{T} -> Void
# move!: Var{T} x Map{T, V} -> Var{V}
# measure!: Var{T} x Measure{T, F} -> Dist{F}
# _unregister!: Var{V} -> Void

 
# implementation


include("spaces/finite.jl")
include("etypes.jl")

# ===================
# Hilbert Space Utilities for Typed Programming
# ===================

CH = Union{Vector, H} # a tree made of Hs

flatten(h ::H) ::Vector{H} = [h]
flatten(ch ::Vector) ::Vector{H} = vcat([flatten(f) for f in ch]...)

conj(ch ::Vector) = [conj(h) for h in ch]

function create_hilbert(t ::Type) ::CH
    if decompose(t) == t
        H(size(t)[1])
    else
        [create_hilbert(x) for x in decompose(t)]
    end
end

function global_hilbert(t ::Type, hash ::UInt = objectid(t)) ::CH
    if decompose(t) == t
        H(HASHPRIME * hash, size(t)[1])
    else
        [global_hilbert(x, HASHPRIME * hash + i) for (i, x) in enumerate(decompose(t))]
    end
end


@testset "compound hilbert spaces" begin
    @test global_hilbert(Bool) == global_hilbert(Bool)
    h = create_hilbert(Tuple{Bool, Bool}) ::CH
    @test length(h) == 2
    @test dim(h[1]) == dim(h[2]) == 2
end


# =====================
# Variables and Storage
# =====================

abstract type Q{T} end

struct Var{T} <: Q{T}
    hs ::CH
end

var(t ::T) where {T} = register!(val(t)) ::Q{T}
var(t ::Set{T}) where {T} = register!(val(t)) ::Q{T}
# f : t -> complex
var(t ::Type{T}, f ::Function) where {T} = register!(val(t, f)) ::Q{T}

blocks = Vector{DenseVector}()
registry = Dict{H, Int}()

function register!(val::Val{T}) ::Q{T} where {T}
    hs = create_hilbert(T)
    b = morph(val.t, space(val.t), flatten(hs))
    push!(blocks, b)
    Var{T}(hs)
end

@testset "variable registration" begin
    v1 = var(true)
    v2 = var(Set([(true, false), (false, true)]))
    @test isa(v1, Q{Bool})
    @test isa(v2, Q{Tuple{Bool, Bool}})
end

# ===================
# Views, like references
# ===================

struct View{T} <: Q{T}
    src ::Vector{H}
    map ::DenseVector # bound to hilbert spaces
    hs ::CH
end

promote_rule(::Type{View{T}}, ::Type{Var{T}}) where {T} = View{T}

convert(::Type{View}, v ::Var) =
     View(flatten(v.hs), convert(DenseVector, 1), v.hs) ::View

function compose_impl(::Type{T}, vs ::Var...) ::Var{T} where {T}
    @assert all([isempty(intersect(flatten(x.hs), flatten(y.hs)))
        for (i, x) in enumerate(vs) for y in vs[i+1:end]])
    Var{T}([v.hs for v in vs])
end

function compose(::Type{T}, qs ::Q...) ::Q{T} where {T}
    compose_impl(T, promote(qs...)...)
end

function compose_impl(::Type{T}, vs ::View...) ::View{T} where {T}
    @assert all([isempty(intersect(x.src, y.src))
        for (i, x) in enumerate(vs) for y in vs[i+1:end]])
    View{T}(vcat([v.src for v in vs]...), prod([v.map for v in vs]), [v.hs for v in vs])
end

function decompose(v ::Var{T}) ::Tuple where {T}
    Tuple([Var{t}(h) for (h, t) in zip(v.hs, decompose(T))])
end

function decompose(v ::View{T}) ::Tuple{View} where {T}
    Tuple([all(x -> x in v.src, flatten(h)) ?
            Var{t}(h) :
            View{t}(filter(x -> !(x in flatten(h)), v.src), v.map, h)
        for (h, t) in zip(v.hs, decompose(T))])
end

function reinterpret(v ::Q{T}, m ::Map{T, V}) ::View{V} where {T, V}
    v = convert(View, v)
    dst = create_hilbert(V)
    View{T}(v.src, m.t(flatten(v.hs), flatten(dst)) * v.map, dst)
end

function cast(::Type{T}, v ::Q{V}) ::Q{T} where {T, V}
    # TODO reinterpret cast
end

@testset "three type of view-transforms" begin
    v1 = var(false)
    v2 = var(false)
    v1v2 = compose(Tuple{Bool, Bool}, v1, v2)
    @test (v1, v2) == decompose(v1v2)
    # TODO test reinterpret
end

function entangle!(hs ::Vector{H})
    source = registry[hs[1]]
    for h in hs[1:end]
        if registry[h] != source
            block[source] = block[source] * block[registry[h]]
            registry[h] = source
        end
    end
end

function apply!(v ::Var{T}, m ::Map{T, T}) where {T}
    site = entangle!(flatten(v.hs))
    blocks[site] = m.t(flatten(v.hs), flatten(v.hs)) * blocks[site]
end

function apply!(v ::View{T}, m ::Map{T, T}) where {T}
    site = entangle!(v.src)
    blocks[site] = inv(v.map) * m.t(flatten(v.hs), flatten(v.hs)) * v.map * blocks[site]
    @assert pnorm(blocks[site]) == 1 # upto an small error?
end

# TODO
# moving view is unallowed (?)
# so what about immutable viewing?
function move!(v ::Var{T}, m ::Map{T, V}) ::Var{T} where {T, V}
    site = entangle!(flatten(v.hs))
    src = flatten(v.hs)
    dst = create_hilbert(V)
    for h in src
        delete!(registry, h)
    end
    for h in flatten(dst)
        registry[h] = site
    end
    blocks[site] = m.t(src, flatten(dst)) * blocks[site]
    @assert pnorm(blocks[site]) == 1 # upto an small error?
    Var{V}(dst)
end

struct Op{T} # positive operator on T
    t ::DenseVector
end

struct Measure{T, F}
    d ::Dict{Op{T}, F}
end

op(v ::T) where {T} = op(convert(Val{T}, v))
op(v ::Set{T}) where {T} = op(convert(Val{T}, v))
op(v ::Val{T}) where{T} = Op{T}(v.t * conj(v.t))
op(v ::Map{Val{T} => Real}) where {T}
# T, Val{T} -> Op{T}
# Set{T} -> Op{T}
# (T1, T2 -> Complex) -> Op?

# default_measurement(::Type{T}) = # default basis measurement

function measure!(v ::Q{T}, m ::Measure{T, F}) :: F where {T, F}
end

function observe!(v ::Q{T}) ::T
    site =
end
# WARNING fock and schrodingers are unsupported

function _unregister!(v ::Var{T}) where {T}
    # disentangle (measure)
    # remove block site
    # remove from registry
end

#=
Q{Bool} qubit = q(folan)
Q{Bool} view(q, u(unitary))
location, momentum = q(folan, views=(u(location), u(momentum)))
e(qubit)
measure(q)
register(qubit);
=#
