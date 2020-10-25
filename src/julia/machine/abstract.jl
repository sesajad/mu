
# ===================
# R-Values
# ===================

# pure model
abstract type PureState{T} end 	# T -> Complex 		# norm vector
(m ::Val)(h ::Vector{AbsHilbert}) ::AbsVector = error("not implemented")
convert(::PureState{T}, t ::T) = error("not implemented")
convert(::PureState{T}, t ::Set{T}) = error("not implemented")

abstract type Map{T, V} end 	# V -> Val{T} 	# semi-unitary
(m ::Map)(from :: Vector{AbsHilbert}, to ::Vector{AbsHilbert}) ::AbsVector = error("not implemented")

Evolution{T} = Map{T, T} 	# a special case of map


abstract type Operator{T} end 	# T x T -> Complex	# hermitian
				# PureState{T} -> Real
				# or Set{(PureState, Real)}
Hamiltonian{T} = Operator{T}
abstract type Measure{T, V} end # Set{(Operator, V)}	# operators must be positive

# probabilistic model
abstract type MixedState{T} end # Set{(PureState, Real)}
convert(::MixedState{T}, t ::PureState{T}) = error("not implemented")

abstract type Channel{T} end 	# MixedState{T} -> MixedState{T}
convert(::Channel{T}, t ::Evolution{T}) = error("not implemented")

abstract type Dist{T} end 	# T -> Real



# =====================
# Variables and Storage
# =====================

# storage model
abstract type Q{T} end
abstract type Var{T} <: Q{T} end
abstract type View{T} <: Q{T} end

promote_rule(::Type{View{T}}, ::Type{Var{T}}) where {T} = View{T}
convert(::Type{View}, v ::Var) = error("not implemented")

# constructor
register!(val::Val{T}) ::Var{T} where {T}

# actions
compose(::Type{T}, qs ::Q...) ::Q{T} where {T} = error("not implemented")
decompose(v ::Q{T}) ::Tuple where {T} = error("not implemented")
cast(::Type{T}, v ::Q{V}) ::Q{T} where {T, V} = error("not implemented")

# views
reinterpret(v ::Q{T}, m ::Map{T, V}) ::View{V} where {T, V} = error("not implemented")

# private modifier functions
_entangle!(hs ::Vector{AbsHilbert}) = error("not implemented")
_unregister!(v ::Var{T}) where {T} = error("not implemented")
_observe!(v ::Q{T}) ::T where {T} = error("not implemented")

# primitive functions
apply!(v ::Var{T}, m ::Evolution{T}) where {T} = error("not implemented")
move!(v ::Var{T}, m ::Map{T, V}) ::Var{T} where {T, V} = error("not implemented")
measure!(v ::Q{T}, m ::Measure{T, F}) :: F where {T, F} = error("not implemented")

