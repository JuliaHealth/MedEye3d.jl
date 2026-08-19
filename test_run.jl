env = copy(ENV)
worker_script_path = joinpath(@__DIR__, "scripts", "ai", "python_worker.py")
cmd = `$(joinpath(@__DIR__, "pyenv", "bin", "python")) -u $worker_script_path`
proc = run(pipeline(setenv(cmd, env), stdout="/tmp/medeye3d_python.log", stderr="/tmp/medeye3d_python.log"), wait=false)
while !process_exited(proc)
    sleep(1)
end
