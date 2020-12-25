module AbstractMachine

using AbstractSpace

# ===================
#      R-Values
# ===================

# pure model
abstract type PureState{T} end 	# T -> Complex 		# norm vector
(m ::PureState)(h ::Vector{AbsHilbert}) ::AbsVector = error("not implemented")
convert(::PureState{T}, t ::T) where {T} = error("not implemented")
convert(::PureState{T}, t ::Set{T}) where {T} = error("not implemented")

abstract type AbsMap{T, V} end 	# V -> PureState{T} 	# semi-unitary
(m ::AbsMap)(from :: Vector{AbsHilbert}, to ::Vector{AbsHilbert}) ::AbsVector = error("not implemented")

Evolution{T} = AbsMap{T, T} 	# a special case of map


abstract type Operator{T} end 	# T x T -> Complex	# hermitian
				# PureState{T} -> Real
				# or Set{(PureState, Real)}
Hamiltonian{T} = Operator{T}
abstract type Measure{T, V} end # Set{(Operator, V)}	# operators must be positive

# probabilistic model
abstract type MixedState{T} end # Set{(PureState, Real)}
convert(::MixedState{T}, t ::PureState{T}) where {T} = error("not implemented")

abstract type Channel{T} end 	# MixedState{T} -> MixedState{T}
convert(::Channel{T}, t ::Evolution{T}) where {T} = error("not implemented")

abstract type Dist{T} end 	# T -> Real



# =====================
#  Variables and Views
# =====================

# storage model
abstract type Q{T} end
abstract type AbsVar{T} <: Q{T} end
abstract type AbsView{T} <: Q{T} end

promote_rule(::Type{AbsView{T}}, ::Type{AbsVar{T}}) where {T} = AbsView{T}
convert(::Type{AbsView}, v ::AbsVar) = error("not implemented")

# constructor
register!(val::PureState{T}) where {T} = error("not implemented")  ::AbsVar{T}

# actions
compose(::Type{T}, qs ::Q...) where {T} = error("not implemented") ::Q{T}
decompose(v ::Q{T}) where {T} = error("not implemented") ::Tuple
cast(::Type{T}, v ::Q{V}) where {T, V} = error("not implemented") ::Q{T}

# views
reinterpret(v ::Q{T}, m ::AbsMap{T, V}) where {T, V} = error("not implemented") ::AbsView{V}

# private modifier functions
_entangle!(hs ::Vector{AbsHilbert}) = error("not implemented")
_unregister!(v ::AbsVar{T}) where {T} = error("not implemented")
_observe!(v ::Q{T}) where {T} = error("not implemented") ::T

# primitive functions
apply!(v ::AbsVar{T}, m ::Evolution{T}) where {T} = error("not implemented")
move!(v ::AbsVar{T}, m ::AbsMap{T, V}) where {T, V} = error("not implemented") ::AbsVar{T}
measure!(v ::Q{T}, m ::Measure{T, F}) where {T, F} = error("not implemented") :: F

export PureState, AbsMap, AbsVar, AbsView, Operator, Evolution, Hamiltonian, Q

end
