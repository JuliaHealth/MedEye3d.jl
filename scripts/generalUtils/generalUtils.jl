module generalUtils
function replace_(rep,expr::Expr)
    i = findfirst(x->x==:_,expr.args)
    if i == 0
        # recurse on subexprs one at a time
        # subexprs = expr.args[]
        error("dun goofed")
    else
        return Expr(expr.head,
                    [expr.args[1:i-1]
                     rep
                     expr.args[i+1:end]]...)
    end
end

macro _(block)
    valid = filter(x->typeof(x) != LineNumberNode &&
                   (typeof(x) != Expr
                    || x.head != :line),block.args)
    foldl(replace_,valid)
end

@_ begin
    3
    +(_,4)
    _^2
end # => 49





end #module