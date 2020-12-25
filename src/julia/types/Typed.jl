module Typed

import Base:conj
using Test

using AbstractSpace

# ===================
# Hilbert Space Utilities for Typed Programming
# ===================

CH = Union{Vector, AbsHilbert} # a tree made of Hs

flatten(h ::T) where {T <: AbsHilbert} = T[h] ::Vector{T}
flatten(ch ::Vector) = vcat([flatten(f) for f in ch]...)

conj(ch ::Vector) = [conj(h) for h in ch]


using FiniteSpace
using Types

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

@testset "Typed:: compound hilbert spaces" begin
    @test global_hilbert(Bool) == global_hilbert(Bool)
    h = create_hilbert(Tuple{Bool, Bool}) ::CH
    @test length(h) == 2
    @test dim(h[1]) == dim(h[2]) == 2
end

export Types
export CH, flatten, conj, create_hilbert, global_hilbert

end
