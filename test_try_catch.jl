using Dates
try
    open("/tmp/medeye3d_errors.log", "a") do f
        println(f, "$(Dates.now()) [reactToAddAutoPet] ERROR: Test msg")
    end
    println("File written successfully")
catch e
    println("Error: $e")
end
