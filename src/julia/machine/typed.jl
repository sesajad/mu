
# ===================
# Hilbert Space Utilities for Typed Programming
# ===================

CH = Union{Vector, AbsHilbert} # a tree made of Hs

flatten(h ::AbsHilbert) ::Vector{AbsHilbert} = [h]
flatten(ch ::Vector) ::Vector{AbsHilbert} = vcat([flatten(f) for f in ch]...)

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

