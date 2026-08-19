using GLMakie
fig = Figure()
status = Observable("Ready")
Label(fig[1, 1], @lift(string($status)))

try
    println("Setting status to emoji...")
    status[] = "🔄 Processing"
    println("Success! (This shouldn't print in devcontainer)")
catch e
    println("CAUGHT ERROR:")
    println(sprint(showerror, e, catch_backtrace()))
end
