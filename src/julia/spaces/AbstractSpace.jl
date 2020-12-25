module AbstractSpace

import Base:*, ==, +, -, /, conj, convert

const TT = ComplexF64

# ===================
#    Hilbert Space
# ===================

# this structure is the unqiue identifier of each hilbert space
# note that this is an internal structure.

abstract type AbsHilbert end

hash(h ::AbsHilbert) ::UInt = error("not implemented")
==(a ::AbsHilbert, b ::AbsHilbert) = error("not implemented")


dual(h ::AbsHilbert) ::AbsHilbert = error("not implemented")
idual(h ::AbsHilbert) ::AbsHilbert = error("not implemented")
conj(h ::AbsHilbert) ::AbsHilbert = error("not implemented")
# assumming h.duality is in {0, 1}


# ===================
#  Vectors
# ===================

abstract type AbsVector end

space(t ::AbsVector) ::Vector{AbsHilbert} = error("not implemented")
order(t ::AbsVector) ::Int = length(space)

==(a ::AbsVector, b ::AbsVector) ::Bool = error("not implemented")
isapprox(a ::AbsVector, b ::AbsVector) ::Bool = error("not implemented")

# field operations
*(a ::AbsVector, b ::AbsVector) ::AbsVector = error("not implemented")
+(a ::AbsVector, b ::AbsVector) ::AbsVector = error("not implemented")
-(a ::AbsVector, b ::AbsVector) ::AbsVector = a + (-1) * b
*(a ::T, b ::Number) where {T <: AbsVector} = (a * convert(T, b)) ::T
*(b ::Number, a ::AbsVector) ::AbsVector = a * b
/(a ::AbsVector, b ::Number) ::AbsVector = a * (1 / b)

# space operations
dual(t ::AbsVector) ::AbsVector = error("not implemented")
idual(t ::AbsVector) ::AbsVector = error("not implemented")
conj(t ::AbsVector) ::AbsVector = error("not implemented")
morph(t ::AbsVector, from ::Vector{AbsHilbert}, to ::Vector{AbsHilbert}) ::AbsVector = error("not implemented")

# numeric conversions
convert(::Type{AbsVector}, n ::Number) ::AbsVector = error("not implemented")
convert(::Type{Number}, t ::AbsVector) ::Number = error("not implemented")

# mathematical functions
pnorm(t ::AbsVector, p ::Int=2) ::Number = error("not implemented")
tr(b ::AbsVector, hs ::Vector{AbsHilbert}) ::AbsVector = error("not implemented")


# exports
export TT
export AbsHilbert, hash, ==, dual, idual, conj
export AbsVector, space, order, ==, +, -, *, /, dual, idual, conj, morph, convert, pnorm, tr

end
