path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)
code = """
    # Helper to trigger observables
    function trigger_obs(name)
        try
            LMW = MEH._get_lmw()
            if LMW !== nothing
                obs = getfield(LMW, :_lmw_observables)
                if haskey(obs, name)
                    if typeof(obs[name][]) == Int
                        obs[name][] += 1
                    else
                        obs[name][] = obj
                    end
                end
            end
        catch; end
    end
"""
content = replace(content, """
    # Helper to trigger observables
    function trigger_obs(name)
        try
            LMW = MEH._get_lmw()
            if LMW !== nothing
                obs = getfield(LMW, :_lmw_observables)
                if haskey(obs, name)
                    obs[name][] += 1
                end
            end
        catch; end
    end
""" => code)
write(path, content)
println("Patched trigger_obs")
