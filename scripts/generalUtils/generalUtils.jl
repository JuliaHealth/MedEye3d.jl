module generalUtils
export replace_
export vecvec_to_matrix

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


```@doc
changing a vector of vectors to the matrix
taken from https://stackoverflow.com/questions/63892334/using-broadcasting-julia-for-converting-vector-of-vectors-to-matrices
```
function vecvec_to_matrix(vecvec)
    dim1 = length(vecvec)
    dim2 = length(vecvec[1])
    my_array = zeros(Int64, dim1, dim2)
    for i in 1:dim1
        for j in 1:dim2
            my_array[i,j] = vecvec[i][j]
        end
    end
    return my_array
  end
  





end #module