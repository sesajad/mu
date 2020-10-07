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
abstract type MixedState{T} end # Set{(PureState, Real)}

abstract type Channel{T} end 	# MixedState{T} -> MixedState{T}

abstract type Dist{T} end 	# T -> Real

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


abstract type Q{T} end

struct QSimpleVar{T} <: Q{T}
    block ::Int
    h ::H
end

struct QComposite{T} <: Q{T}
    qs ::Vector{Q}
end

hs(q ::QSimpleVar) = [q.h]
hs(q ::QComposite) = [h for qq in q.qs for h in hs(qq)]

function var(t ::T) ::Q{T} where {T}
    register!(T, [tt == t ? 1 : 0 for tt in iter(T)])
end

function var(t ::Set{T}) ::Q{T} where {T}
    register!(T, [tt in t ? 1 / sqrt(length(t)) : 0 for tt in iter(T)])
end

function var(f ::Function) ::Q{T} where {T} # t -> complex
    register!(T, [f(tt) for tt in iter(T)])
end

blocks = Vector{Tensor}()
registry = Vector{QSimpleVar}()

function register_address!(t ::Type, block ::Int)
    if decompose(t) == t
        var = QSimpleVar{t}(block, H(size(t)[1]))
        push!(registry, var)
        var
    else
        QComposite{t}([register_address!(tt, block) for tt in decompose(t)])
    end
end

function register!(t ::Type{T}, val::Array) ::Q{T} where {T}
    @assert size(t) == size(val)
    address = register_address!(t, length(blocks))
    b = Tensor(val, hs(address))
    push!(blocks, b)

    address
end

@testset "variable registration" begin
    v1 ::Q{Bool} = var(true)
    v2 ::Q{Tuple{Bool, Bool}} = var(Set([(true, false), (false, true)]))
    @test isa(v1, QSimpleVar{Bool})
    @test isa(v2, QComposite{Tuple{Bool, Bool}})
end

# or for any other composition

function view(qs ::Q...) ::Q{Tuple}
    # TODO: check qs are disjoint
    params = [typeof(q).parameters[1] for q in qs]
    QComposite{Tuple{params...}}(qs)
end

function view(qs ::Q{Tuple}) ::Tuple
    (q.qs...,)
end

struct Map{T, V}
    val ::Tensor
    src ::Vector{H}
    dst ::Vector{H}
end

function map(f ::Function) where {T, V} # f: T -> Q{V}
    src = [H(s) for s in size(T)]
    dst = [H(s) for s in size(V)]
end


# map: Function{T, V} -> Map{T, V} (tool)
# map: Function{T, VirtualMixedState{V}} -> Map{T, V}


# TODO mapping
#struct QMapped{T} <: Q{T}
#    ref ::Q
#    map ::Map{T}
#end
#function view(q ::Q{T}, map ::QMap{T, V}) ::Q{V} where {T, V}
#    hv = H(size(V))
#    Q(map(from=q.hs, to=hv) * q.premap, hv)
#end

#=


struct QVal{T} <: PureState{T}
    vec ::Tensor                           # must match dimension
    
end


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
