ex = :(1 + 2)

typeof(ex)

dump(ex)

eval(ex)

macro sayhello(name)
    return :(println("Hello ", $name))
end

@sayhello("Skater")

dump(@macroexpand @sayhello("Skater"))

ex = :(begin x = 1; y = 2 end)
ex.args
filter(x -> !(x isa LineNumberNode), ex.args)   # just the "real" lines


macro mystruct(name, body)
    
    body.head == :block || error("Expected a block of expressions")
    filter!(x -> !(x isa LineNumberNode), body.args)

    fields = body.args
    esc(quote
        struct $name
            $(fields...)
        end
    end)
end


function _resolve_size(arg,dn::Set{Symbol})
    (arg isa Symbol && arg ∈ dn) ? :(data.$arg) : arg
end
_lines(b::Expr) = filter(x -> !(x isa LineNumberNode), b.args)

function _parse_data(block::Expr)
    fields = Expr[]
    names = Symbol[]
    for line in _lines(block)
        line isa Expr && line.head == :(::) || continue
        var = line.args[1]
        typespec = line.args[2]
        var isa Symbol || @warn "Expected a Symbol, got $(typeof(var)), ignoring." continue
        push!(fields, :($var::$(_dsl_to_julia_type(typespec))))
        push!(names, var)
    end
    fields,names
end

function _dsl_to_julia_type(spec)
    spec isa Symbol && return spec
    spec isa Expr && spec.head == :curly && return spec

    spec isa Expr && spec.head == :call && return spec.args[1]
    return spec
end


macro specsheet(model_name::Symbol, body::Expr)
    body.head == :block || error("@specsheet expects begin...end block")
    data_blk = params_blk = xform_blk = model_blk = mcmc_blk = nothing
    for expr in body.args
        expr isa Expr && expr.head == :macrocall || continue
        sym = expr.args[1]
        blk = last(filter -> a isa Expr, expr.args)

        sym == Symbol("@data") && (data_blk = blk)
        sym == Symbol("@params") && (params_blk = blk)
        sym == Symbol("@xform") && (xform_blk = blk)
        sym == Symbol("@logjoint") && (model_blk = blk)
        sym == Symbol("@mcmc") && (mcmc_blk = blk)
    end

    data_blk !== nothing || error("Missing @data block")
    params_blk !== nothing || error("Missing @params block")
    xform_blk !== nothing || error("Missing @xform block")
    model_blk !== nothing || error("Missing @logjoint block")
    mcmc_blk !== nothing || error("Missing @mcmc block")

    data_fields, data_names = _parse_data(data_blk)
    dn = Set(data_names)
end

ex = quote
    @data begin
        x::Int
        y::Float64
    end
    @params begin
        alpha = param(Vector{Float64}, p)
        gamma = param(Vector{Float64}, p, lower = 0.0, upper = 1.0)

        beta::Float64
    end
end

dump(_lines(ex))



"""Map (lo, hi) keyword values to the appropriate Constraint."""
function _make_constraint(lo, hi)
    isnothing(lo) && isnothing(hi)  && return IdentityConstraint()
    !isnothing(lo) && isnothing(hi) && return LowerBounded(lo)
    isnothing(lo) && !isnothing(hi) && return UpperBounded(hi)
    return Bounded(lo, hi)
end

struct _ParamSpec
    name::Symbol
    constraint::Constraint
    container::Symbol       # :scalar | :vector | :matrix
    sizes::Vector{Any}
end


function _parse_params(block::Expr,data_names::Set{Symbol})
    specs = _ParamSpec[]
    for line in _lines(block)
        line isa Expr || continue

        if line.head == :(::)
            var = line.args[1]
            T = line.args[2]
            var isa Symbol || error(
                "@params: '$var::$T' — bare annotations only support Float64. " *
                "For vectors and matrices use param(Vector{Float64}, n, ...)")
            push!(specs, _ParamSpec(var, IdentityConstraint(), :scalar, []))

        elseif line.head == :(=)
            var = line.args[1]
            rhs = line.args[2]
            rhs isa Expr && rhs.head == :call && rhs.args[1] == :param || error(
                "@params: '$var = $rhs' — expected a call to param(...)")
            push!(specs, _param_to_spec(var, rhs.args[2:end], data_names))
        end
    end
    specs 
end

"""Convert `param(T, sizes...; lower=…, upper=…)` to _ParamSpec."""
function _param_to_spec(name::Symbol, args, dn::Set{Symbol})
    isempty(args) && error("@params: param() requires a type as its first argument")

    T    = args[1]
    rest = args[2:end]
    sz_args = [a for a in rest if !(a isa Expr && a.head == :kw)]
    kw_args = [a for a in rest if   a isa Expr && a.head == :kw]

    # Extract bounds from keywords
    lo = hi = nothing
    for a in kw_args
        a.args[1] == :lower && (lo = a.args[2])
        a.args[1] == :upper && (hi = a.args[2])
    end

    constraint = _make_constraint(lo, hi)

    # Scalar
    if T == :Float64
        isempty(sz_args) || error("@params: param(Float64) takes no positional size arguments")
        return _ParamSpec(name, constraint, :scalar, [])
    end

    # Vector or Matrix
    if T isa Expr && T.head == :curly
        base = T.args[1]
        elem = T.args[2]
        if base == :Vector && elem == :Float64
            length(sz_args) == 1 || error("@params: param(Vector{Float64}, n) takes one size argument")
            return _ParamSpec(name, constraint, :vector, [_resolve_size(sz_args[1], dn)])
        elseif base == :Matrix && elem == :Float64
            length(sz_args) == 2 || error("@params: param(Matrix{Float64}, n, m) takes two size arguments")
            constraint isa IdentityConstraint || error("@params: bounded matrices not supported")
            sz1 = _resolve_size(sz_args[1], dn)
            sz2 = _resolve_size(sz_args[2], dn)
            return _ParamSpec(name, constraint, :matrix, [sz1, sz2])
        end
    end

    error("@params: unsupported type in param(): $T")
end

lines = filter(x -> !(x isa LineNumberNode), ex.args)

dump(lines[2].args[3].args[2])

