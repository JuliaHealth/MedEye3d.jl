include("src/display/InferenceClient.jl")
using .InferenceClient
InferenceClient.start_python_worker(joinpath(@__DIR__, "scripts", "ai", "python_worker.py"))
