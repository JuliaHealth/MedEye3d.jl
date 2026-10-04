using JSON
try
    v = tryparse(Float32, 0.0)
    println("Success")
catch e
    println("Error: ", typeof(e))
end
