# f: T -> Bool
function grover_imperative(v ::Val{T}, f ::Function) where {T} ::T
    q = Q(v)
    decreate = pseudoinv(convert(Map{Unit, T}, v))
    for _ in 1:10
        if f(q)
            q = q ~ 1 # or =~ 1
        if decreate(q).type == unit
            q = q ~ 1
    end
    return measure q
end

### COMPILED to

function grover_imperative(v ::Val{T}, f ::Function) where {T} ::T
    q = register!(v)
    decreate = pseudoinv(convert(Map{Unit, T}, v))
    for _ in 1:10
        apply!(qiffi(map(f), -identity, identity, map(f)), q)
        apply!(qiffi(decreate)

    end
    return measure q
end


function grover_functional(v ::Val{T}, f ::Function) where {T} ::T
    decreate = pseudoinv(convert(Map{Unit, T}, v))
    grover_step = (x -> f(x) ? x ~ 1 : x).then(y -> decreate(y).type == 1 ? y ~ 1 : y)

    grover_rec = (n, s) -> n == 0 : s ? grover_rec(n - 1, grover_step(s))

    return measure grover_rec(10, v)
end

### COMPILED to

function grover_functional(v ::Val{T}, f ::Function) where {T} ::T

end
