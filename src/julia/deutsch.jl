# pure model
abstract type PureState{T} end 	# T -> Complex 		# norm vector

abstract type UnitaryMap{T, V} end 	# V -> PureState{T} 	# unitary
abstract type Evolution{T} <: UnitaryMap{T, T} end 	# a special case of UnitaryMap


abstract type Operator{T} end 	# T x T -> Complex	# hermitian
				# PureState{T} -> Real  
				# or Set{(PureState, Real)}
abstract type Hamiltonian{T} <: Operator{T} end
abstract type Measure{T, V} end # Set{(Operator, V)}	# operators must be positive

# probabilistic model
abstract type MixedState{T} end # Set{(PureState, Real)}

abstract type Channel{T} end 	# MixedState{T} -> MixedState{T}

abstract type Dist{T} end 	# T -> Real

# single-var functions
# cast: T -> PureState
# cast: PureState{T} -> MixedState{T}
# cast: Evolution{T} -> Channel{T}

# composition functions

# storage, an mutable interface
abstract type VirtualPureState{T} end
abstract type VirtualMixedState{T} end

# view: VirtualMixedState{T} x Basis{T, V} -> VirtualMixedState{V}
# view: VirtualMixedState{NTuple{N, T}} -> NTuple{VirtualMixedState{T}}

# Modifiers of global state

# register!: MixedState{T} -> VirtualMixedState{T}
# entangle!: NTuple{VirtualMixedState{T}} -> VirtualMixedState{NTuple{N, T}} 

# erase!: VirtualMixedState{V} -> Void
# apply!: VirtualMixedState{T} x Channel{T} -> Void
# move!: VirtualMixedState{T} x Basis{T, V} -> VirtualMixedState{V}
# measure!: VirtualMixedState{T} x Measure{T, F} -> Dist{F}

 
# implementation

# Note that T must be cartesian-enumeratable and known-sized

# index(t::T), dim(T)
include("hilbert.jl")

struct QVal{T} <: PureState{T}
    vec ::Tensor                           # must match dimension
    
end

struct QMap{T, V} <: Basis{T, V}
    matrix ::Tensor
end

struct QOp{T} <: QMap{T, T}, Evolution{T}, Operator{T}
end

struct QMeasure{T, F}
    set ::Vector{Tuple{Operator, F}}
end

type Q{T} <: VirtualPureState{T} end
mutable struct VariableAddress{T} <: Q{T]
    block ::Int8
    dim ::Range
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

function q(t ::T) ::Q{T} where {T}
    register!(QVal(ket(index(t), n=size(T))))
end

function q(t ::Set{T}) ::Q{T} where {T}
    register!(QVal(sum([ket(index(tt), n=size(T)) for tt in t])))
end

function q(t ::Function) ::Q{T} where {T} # t -> complex
    register!(QVal(sum(map(i -> t(value{T}(i)) * ket(i, n=size(T)), 1:10))))
end

function register!(val::QVal{T}) ::Q{T} where {T} 
    block.append(val)
    address = VariableAddress(block=length(block), dim=1)
    registry.append(address)
    address
end

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


#=
Q{Bool} qubit = q(folan)
Q{Bool} view(q, u(unitary))
location, momentum = q(folan, views=(u(location), u(momentum)))
e(qubit)
measure(q)
register(qubit);
=#
