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

# single-var functions
# cast: T -> PureState{T}
# cast: Set{T} -> PureState{T}
# cast: PureState{T} -> MixedState{T}
# cast: Evolution{T} -> Channel{T}

# composition functions

# storage, an mutable interface
# abstract type VirtualPureState{T} end
# abstract type VirtualMixedState{T} end

# constructors
# var: T -> VirtualMixedState{T} (tool)
# var: Set{T} -> VirtualMixedState{T} (tool)
# var: Function{T} -> VirtualMixedState{T}
# map: Function{T, V} -> Map{T, V} (tool)
# map: Function{T, VirtualMixedState{V}} -> Map{T, V}

# functions
# let: VirtualMixedState{T} x Map{T, V} -> VirtualMixedState{V}
# view: VirtualMixedState{NTuple{N, T}} -> NTuple{VirtualMixedState{T}}
# view: NTuple{VirtualMixedState{T}} -> VirtualMixedState{NTuple{N, T}}

# Modifiers of global state

# _register!: MixedState{T} -> VirtualMixedState{T}
# _entangle!: list of blocks to entangle

# erase!: VirtualMixedState{V} -> Void
# apply!: VirtualMixedState{T} x Channel{T} -> Void
# move!: VirtualMixedState{T} x Map{T, V} -> VirtualMixedState{V}
# measure!: VirtualMixedState{T} x Measure{T, F} -> Dist{F}

 
# implementation


include("hilbert.jl")
include("etypes.jl")

# ===================
# Hilbert Space Utilities for Typed Programming
# ===================

CH = Union{Vector, H} # a tree made of Hs

flatten(h ::H) ::Vector{H} = [h]
flatten(ch ::Vector) ::Vector{H} = vcat([flatten(f) for f in ch]...)

function create_hilbert(t ::Type) ::CH
    if decompose(t) == t
        H(size(t)[1])
    else
        [create_hilbert(x) for x in decompose(t)]
    end
end

global_hilbert(t ::Type) ::Vector{H} = [H(objectid(t) * HASHPRIME + i, s) for (i, s) in enumerate(size(t))]

@testset "compound hilbert spaces" begin
    @test global_hilbert(Bool) == global_hilbert(Bool)
    h = create_hilbert(Tuple{Bool, Bool}) ::CH
    @test length(h) == 2
    @test dim(h[1]) == dim(h[2]) == 2
end

struct Val{T}
    t ::Tensor
end

# f : T -> Complex
val(t ::Type{T}, f ::Function) where {T} =
    Val{T}(Tensor([f(tt) for tt in iter(T)], global_hilbert(T))) ::Val{T}
val(t ::Set{T}) where {T} = val(T, tt -> ((tt in t) ? (1 / sqrt(length(t))) : 0)) ::Val{T}
val(t ::T)  where {T} = val(Set{T}([t])) ::Val{T}

+(a ::Val{T}, b ::Val{T}) where {T} = Val{T}(a.t + b.t) ::Val{T}
-(a ::Val{T}, b ::Val{T}) where {T} = (a + (-1) * b) ::Val{T}
*(a ::Val{T}, b ::Number) where {T} = Val{T}(a.t * b) ::Val{T}
*(b ::Number, a ::Val{T}) where {T} = Val{T}(a.t * b) ::Val{T}
/(a ::Val{T}, b ::Number) where {T} = Val{T}(a.t / b) ::Val{T}
normalize(v ::Val{T}) where {T} = v / pnorm(v.t)

@testset "values (aka rvalues)" begin
    v1 = val(false)
    v2 = val(Set([false, true]))
    v3 = val(true)
    @test typeof(v1) == typeof(v2) == typeof(v3) == Val{Bool}
    @test normalize(v1 + v3).t == v2.t
    @test convert(Number, conj(v1.t) * v2.t) == 1 / sqrt(2)
end

struct Map{T, V}
    t ::Tensor
end

function map(p::Pair, f ::Function) # f: T -> Val{V}
    T, V = p
    tensor = sum([f(tt).t * conj(val(tt).t) for tt in iter(T)])
    Map{T, V}(tensor)
end

function (m ::Map)(from :: Vector{H}, to ::Vector{H}) ::Tensor
    t = morph(m.t, [dual(h) for h in global_hilbert(T)], [dual(h) for h in from])
    t = morph(t, global_hilbert(V), to)
end

@testset "maps (aka lambdas)" begin
    f = val(false)
    t = val(true)
    hadamard = map(Bool => Bool, x -> (val(true) + (x ? -1 : 1) * val(false)) / sqrt(2))
    @test convert(Number, conj(t.t) * hadamard.t * f.t) == 1 / sqrt(2)
    @test convert(Number, conj(f.t) * hadamard.t * f.t) == 1 / sqrt(2)
end

abstract type Q{T} end

struct Var{T} <: Q{T}
    hs ::CH
end

var(t ::T) where {T} = register!(val(t)) ::Q{T}
var(t ::Set{T}) where {T} = register!(val(t)) ::Q{T}
# f : t -> complex
var(t ::Type{T}, f ::Function) where {T} = register!(val(t, f)) ::Q{T}

blocks = Vector{Tensor}()
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

struct View{T} <: Q{T}
    src ::Vector{H}
    map ::Tensor # bound to hilbert spaces
    hs ::CH
end

promote_rule(::Type{View{T}}, ::Type{Var{T}}) where {T} = View{T}

convert(::Type{View}, v ::Var) =
     View(flatten(v.hs), convert(Tensor, 1), v.hs) ::View

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

function apply(q ::Var{T}, m ::Map{T, T}) where {T}
    site = entangle!(flatten(q.hs))
    blocks[site] = m.t(flatten(q.hs), flatten(q.hs)) * blocks[site]
end

function apply(q ::View{T}, m ::Map{T, T}) where {T}
    # TODO
end

## I left here, then move

# map: Function{T, V} -> Map{T, V} (tool)
# map: Function{T, VirtualMixedState{V}} -> Map{T, V}


#=

struct QOp{T} <: QMap{T, T}, Evolution{T}, Operator{T}
end

struct QMeasure{T, F}
    set ::Vector{Tuple{Operator, F}}
end

type Q{T} <: VirtualPureState{T} end
mutable struct VariableAddress{T} <: Q{T]
    block ::Int8
    h ::H
end
struct CrossAddress{T} <: Q{T}
    refs ::Vector{Q}
end

struct ProjectAddress{T} <: Q{T}
    ref ::Q
    dim ::Set{Int8}
end

struct RotateAddress{T} <: Q{T}
    ref ::Q
    matrix ::Gate
end

blocks = []
registry = []

function entangle!(t ::Set{VariableReferences}) where {T, n}
    blocks = sort([tt.block for tt in t]) 
    # for improving performance sort blocks descending  
    a = blocks[end] # smallest number
    for b in blocks[1:end - 1] # from largest to smallest
        if (typeof(block[a]) == PureState_Vector && 
            typeof(block[b]) == PureState_Vector)
            for r in registry
                if (r.block == b)
                    r.block = a
                    r.dim += dim(block[a].vec)
                elseif (r.block > b)
                    r.block -= 1
                end
            end
            block[a] = block[a] ⊗ b_block
            deleteat!(block, b)
        else
            throw("unimplemented type for entanglement")
        end
    end
end

function view(qs ::Varargs{Q, n}) ::Q{Tuple}
    CrossAddress(refs = [...qs])
end

function view(q ::Q{T}, map ::QMap{T, V}) ::Q{V} where {T, V}
    RotateAddress(ref=q, matrix = map)
end

function view(qs ::Q{Tuple}) ::NTuple{Q, n} where {n}
    start = 1
    res = []
    for t in typeof(qs).parameters[1].parameters:
        res.append(ProjectAddress(ref=qs, range(start, length=length(dim(t)))))
        s += length(dim(t)) 
    (...res)
end

#function u():: 


function unregister!(q ::VarAddress{T}) where {T}
    # reassign the block  (by a measurement-like tracing)
    #=
    block[q.block] = 
    if (block[q.block] is empty) 
        deleteat!(block, q.block)
        then update registry
    else
        update registry
    =#
end

function apply(q ::Vector{Q}, g::Local)
    t = []
    dim = 1
    for s in q 
        case s ::Rotate => apply(q.ref, q.matrix)
        case s ::Dim => 
            g = Local([1:d..., q.dims, :end], g)
            t.append(q.ref)
        case s ::CrossAddress => 
            t.appendAll(q.refs)
        case s ::Variable => nothing
        dim += length(dim(s))
    if (t all variable)
        cast t to Vector{Variable}
    apply(t, g)
    for s in q
        case s ::Rotate => apply(q.ref, q.matrix^-1)
end

function apply!(q ::Vector{Variable}, g::{Gate or Local})
    entangle!(folan)
    block[q.block] = Local(g, vec(q.dim)) * block[q.block]
end


function move!(q ::VariableAddress{T}, map ::QMap{T, V}) ::VariableAddress{V} where {T, V}
    apply!(q, map)
    v = VariableAddress{V}(q.block, q.dim)
    q.block = -1
    q.dim = -1
    v
end

function measure!(q ::Q{T}, Measure{T, F}) ::F
     bra * 
end

Q{Bool} qubit = q(folan)
Q{Bool} view(q, u(unitary))
location, momentum = q(folan, views=(u(location), u(momentum)))
e(qubit)
measure(q)
register(qubit);
=#
