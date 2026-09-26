path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)
new_code = """
                if count > 0
                    m.suv_mean = sum_suv / count
                    m.suv_max = max_suv
                end
            end
            
            # Notify LMW to refresh measurement list
            try
                LMW = MEH._get_lmw()
                if LMW !== nothing
                    obs = getfield(LMW, :_lmw_observables)
                    if haskey(obs, :obs_refresh_measurements)
                        obs[:obs_refresh_measurements][] += 1
                    end
                end
            catch; end
        end
    end
end
"""
content = replace(content, """
                if count > 0
                    m.suv_mean = sum_suv / count
                    m.suv_max = max_suv
                end
            end
        end
    end
end
""" => new_code)
write(path, content)
