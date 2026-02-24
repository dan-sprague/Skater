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
