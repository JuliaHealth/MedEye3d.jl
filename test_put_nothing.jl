try
    put!(nothing, 1)
catch e
    println("Error: ", typeof(e))
end
