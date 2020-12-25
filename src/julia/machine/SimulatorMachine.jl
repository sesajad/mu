module SimulatorMachine

using Test

using FiniteSpace
import FiniteSpace:DenseVector
using Typed
using Types
using AbstractMachine
using SimulatorValues
import SimulatorValues:Val


struct Var{T} <: AbsVar{T}
    hs ::CH
end

blocks = Vector{DenseVector}()
registry = Dict{H, Int}()

struct View{T} <: AbsView{T}
    src ::Vector{H}
    map ::DenseVector # bound to hilbert spaces
    hs ::CH
end

convert(::Type{View}, v ::Var) = View(flatten(v.hs), convert(DenseVector, 1), v.hs) ::View

# constructor
function register!(val::Val{T}) ::Var{T} where {T}
    hs = create_hilbert(T)
    b = morph(val.t, space(val.t), flatten(hs))
    push!(blocks, b)
    for h in flatten(hs)
        registry[h] = length(blocks)
    end
    Var{T}(hs)
end

var(t) = register!(val(t))

@testset "variable registration" begin
    v1 = var(true)
    v2 = var(Set([(true, false), (false, true)]))
    @test isa(v1, Var{Bool})
    @test isa(v2, Var{Tuple{Bool, Bool}})
end


# actions

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

cast(::Type{T}, v ::Q{V}) where {T, V} = error("not implemented") ::Q{T}

function cast(::Type{T}, v ::Var{V}) ::Var{T} where {T, V}
    #Var{V}(
    # TODO
end

# views

function reinterpret(v ::Q{T}, m ::Map{T, V}) ::View{V} where {T, V}
    v = convert(View, v)
    dst = create_hilbert(V)
    View{T}(v.src, m.t(flatten(v.hs), flatten(dst)) * v.map, dst)
end

# ===================
# Views, like references
# ===================

@testset "three type of view-transforms" begin
    v1 = var(false)
    v2 = var(false)
    v1v2 = compose(Tuple{Bool, Bool}, v1, v2)
    println(v1v2)
    @test (v1, v2) == decompose(v1v2)
    z = reinterpret(v1, fourier(Bool))
    @test z isa View{Bool}
end

# private modifier functions

function _entangle!(hs ::Vector{H})
    source = registry[hs[1]]
    for h in hs[1:end]
        if registry[h] != source
            block[source] = block[source] * block[registry[h]]
            registry[h] = source
        end
    end
end

function _observe!(v ::Var{T}) ::T
    if decompose(T) == T
        h = first(v.hs)
        p = rand()
        for t in iter(T)
            observed = conj(Val(t)(v.hs)) * block[registry[h]]
            p -= pnorm(observed)
            if (p <= 0)
                if (order(observed) > 1)
                    block[registry[h]] = observed
                    push!(blocks, Val(t)(h))
                    registry[h] = length(blocks)
                else
                    block[registry[h]] = Val(t)(h)
                end
                break
            end
        end
    else
        tuple([observe!(q) for q in decompose(V)])
    end
end
# WARNING fock and schrodingers are unsupported

function _unregister!(v ::Var{T}) where {T}
    _observe!(v)
    for h in hs
        delete!(block, registry[h])
        delete!(registry, h)
    end
end


# primitive functions

function apply!(v ::Var{T}, m ::Map{T, T}) where {T}
    site = entangle!(flatten(v.hs))
    blocks[site] = m.t(flatten(v.hs), flatten(v.hs)) * blocks[site]
end

function apply!(v ::View{T}, m ::Map{T, T}) where {T}
    site = entangle!(v.src)
    blocks[site] = inv(v.map) * m.t(flatten(v.hs), flatten(v.hs)) * v.map * blocks[site]
    @assert pnorm(blocks[site]) == 1 # upto an small error?
end

function move!(v ::Var{T}, m ::Map{T, V}) ::Var{V} where {T, V}
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

# this function will loose view and its reference
function move!(v ::View{T}) ::Var{T} where {T, V}
    site = entangle!(flatten(v.hs))
    blocks[site] = v.map * blocks[site]
    @assert pnorm(blocks[site]) == 1 # upto an small error?
    for h in v.src
        delete!(registry, h)
    end
    for h in flatten(v.hs)
        registry[h] = site
    end
    Var{T}(v.hs)
end


function measure!(v ::Q{T}, m ::Measure{T, F}) :: F where {T, F}
    site = _entangle!(flatten(v.hs))
    p = rand()
    for (o, f) in m.d
            observed = sqrt(f(site)) * block[site]
            p -= pnorm(observed)
            if (p <= 0)
                block[registry[h]] = observed / pnorm(observed)
                break
            end
    end
end



#=
Q{Bool} qubit = q(folan)
Q{Bool} view(q, u(unitary))
location, momentum = q(folan, views=(u(location), u(momentum)))
e(qubit)
measure(q)
register(qubit);
=#

end
