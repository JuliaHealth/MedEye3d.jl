path = "src/display/LesionMetadataWindow.jl"
content = read(path, String)

code = """
    # Observable to trigger list refresh
    _lmw_observables[:obs_refresh_measurements] = Observable{Any}(nothing)

    function _rebuild_measurement_list!(grid, obj)
        # Clear existing content
        for c in collect(grid.content)
            try delete!(grid, c.content) catch; end
        end
        
        if obj === nothing
            return
        end
"""

content = replace(content, """
    # Observable to trigger list refresh
    _lmw_observables[:obs_refresh_measurements] = Observable(0)

    function _rebuild_measurement_list!(grid, stateObjects)
        # Clear existing content
        for c in collect(grid.content)
            try delete!(grid, c.content) catch; end
        end
        
        if isempty(stateObjects)
            return
        end
""" => code)

code2 = """
        # DO NOT filter out active ones, show them all!
        saved_spheres = obj.measurements
        saved_lines = obj.line_measurements
"""
content = replace(content, """
        obj = stateObjects[1].mainForDisplayObjects
        # DO NOT filter out active ones, show them all!
        saved_spheres = obj.measurements
        saved_lines = obj.line_measurements
""" => code2)

code3 = """
    end

    on(_lmw_observables[:obs_refresh_measurements]) do obj
        if obj !== nothing
            _rebuild_measurement_list!(meas_list_grid, obj)
        end
    end

    end_section!(sec_meas)
"""
content = replace(content, """
    end

    end_section!(sec_meas)
""" => code3)

write(path, content)
println("Patched LMW")
