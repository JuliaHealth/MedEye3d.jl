module ShadersAndVerticiesForSupervoxels
using HDF5, Statistics, LinearAlgebra, KernelAbstractions
using ModernGL, GeometryTypes, GLFW
using ..ForDisplayStructs, ..CustomFragShad, ..ModernGlUtil, ..PrepareWindowHelpers, ..DataStructs, ..TextureManag, ..BasicStructs, ..StructsManag, ..DisplayWords
export createAndInitSupervoxelLineShaderProgram, renderSupervoxelLines




"""
Fragment shader for supervoxel lines
"""
function fragShaderSupervoxelLineSrc()
    return """
    #version 330 core
    out vec4 FragColor;




    void main()
    {
        FragColor = vec4(1.0, 1.0, 0.0, 1.0); // Yellow color
    }
    """
end

function createAndInitSupervoxelLineShaderProgram(vertexShader::UInt32)
    fragmentShaderSourceLine = fragShaderSupervoxelLineSrc()
    fsh = """
    $(fragmentShaderSourceLine)
    """
    lineFragmentShader = createShader(fsh, GL_FRAGMENT_SHADER)
    lineShaderProgram = glCreateProgram()
    glAttachShader(lineShaderProgram, lineFragmentShader)
    glAttachShader(lineShaderProgram, vertexShader)
    glLinkProgram(lineShaderProgram)

    return (lineFragmentShader, lineShaderProgram)
end


@kernel function get_cross_section(axis_index::Int, d::Float64, triangle_arr, res_arr)
    index = @index(Global)
    # Determine the axis index
    cross_section_lines = []
    # Iterate through each triangle
    # for i in 1:size(triangle_arr, 1)
    triangle = triangle_arr[index, :, :]
    points = [triangle[j, :] for j in 1:3]

    # Check if the triangle intersects the plane
    distances = [point[axis_index] - d for point in points]
    signs = sign.(distances)

    # if length(unique(signs)) == 1
    #     # All points are on the same side of the plane, no intersection
    #     continue
    # end

    # Compute intersection points
    intersection_points = []
    for j in 1:3
        p1 = points[j]
        p2 = points[mod1(j + 1, 3)]
        d1 = distances[j]
        d2 = distances[mod1(j + 1, 3)]

        if sign(d1) != sign(d2)
            t = d1 / (d1 - d2)
            intersection_point = p1 + t * (p2 - p1)
            push!(intersection_points, intersection_point)
        end
    end

    # if length(intersection_points) == 2
    # distance = norm(intersection_points[1] - intersection_points[2])
    # push!(cross_section_lines, intersection_points)

    if (axis_index == 1)
        ind1 = 2
        ind2 = 3
    end

    if (axis_index == 2)
        ind1 = 1
        ind2 = 3
    end
    if (axis_index == 3)
        ind1 = 1
        ind2 = 2
    end


    base_index = (index - 1) * 2 * 3
    res_arr[base_index+1] = intersection_points[1][ind1]
    res_arr[base_index+2] = intersection_points[1][ind2]
    res_arr[base_index+3] = 1.0
    res_arr[base_index+4] = intersection_points[2][ind1]
    res_arr[base_index+5] = intersection_points[2][ind2]
    res_arr[base_index+6] = 1.0

    # res_arr[base_index+axis_index]=0.0
    # res_arr[base_index+axis_index+3]=0.0



    # end
    # end
    # end

end


function get_example_sv_to_render()
    h5_path_b = "D:/mingw_installation/home/hurtbadly/Downloads/locc.h5"
    fb = h5open(h5_path_b, "r")
    #we want only external triangles so we ignore the sv center
    # we also ignore interpolated variance value here
    tetr_dat = fb["tetr_dat"][:, 2:4, 1:3, 1]


    #given axis and plane we will look for the triangles that points are less then radius times 2 from the plane
    axis = 2
    plane_dist = 41.0
    radiuss = (Float32(4.5), Float32(4.5), Float32(4.5))
    #in order for a triangle to intersect the plane it has to have at least one point on one side of the plane and at least one point on the other side
    bool_ind = Bool.(Bool.((tetr_dat[:, 1, axis] .< (plane_dist)) .* (tetr_dat[:, 2, axis] .> (plane_dist)))
                     .|| Bool.((tetr_dat[:, 2, axis] .< (plane_dist)) .* (tetr_dat[:, 3, axis] .> (plane_dist)))
                     .|| Bool.((tetr_dat[:, 3, axis] .< (plane_dist)) .* (tetr_dat[:, 1, axis] .> (plane_dist)))
    )

    #filter out too long lines
    # bool_ind_b=Bool.( Bool.(abs.((tetr_dat[:, 1, axis]).-(tetr_dat[:, 2, axis] )).<(maximum(radiuss)*2))
    # .&& Bool.(abs.((tetr_dat[:, 2, axis] ).-(tetr_dat[:, 3, axis])).<(maximum(radiuss)*2))
    # .&& Bool.(abs.((tetr_dat[:, 3, axis] ).-(tetr_dat[:, 1, axis])).<(maximum(radiuss)*2) )
    # )
    # #filter out too short lines
    # bool_ind_c=Bool.( Bool.(abs.((tetr_dat[:, 1, axis]).-(tetr_dat[:, 2, axis] )).>(0.01))
    # .&& Bool.(abs.((tetr_dat[:, 2, axis] ).-(tetr_dat[:, 3, axis])).>(0.0))
    # .&& Bool.(abs.((tetr_dat[:, 3, axis] ).-(tetr_dat[:, 1, axis])).>(0.0) )
    # )


    # bool_ind=bool_ind.&&bool_ind_b.&&bool_ind_c
    # #we will only consider the triangles that intersect the plane
    relevant_triangles = Float32.(tetr_dat[bool_ind, :, :])
    # relevant_triangles=tetr_dat

    # # relevant_triangles[:,:,1]
    # relevant_triangles[50,:,:]

    # Int(round(minimum(relevant_triangles[:,:,1])))

    res = Float32.(zeros(size(relevant_triangles, 1) * 2 * 3))
    dev = get_backend(res)
    get_cross_section(dev, 128)(axis, plane_dist, relevant_triangles, res, ndrange=(size(relevant_triangles, 1)))
    KernelAbstractions.synchronize(dev)

    #GETTING TO OPENGL COORDINATE system
    #NOTE : For floating point number calculation please use Float32 instead of Float64 to prevent straight lines

    res = res .+ 1
    res = res ./ 2
    res = res .- minimum(res)
    res = res ./ maximum(res)
    res = res .* Float32(1.6)
    res = res .- 1



    # @info "min" minimum(res)
    # @info "max" maximum(res)


    line_indices = UInt32.(collect(0:(size(relevant_triangles, 1)*16)))
    # line_indices=UInt32.(collect(0:(size(relevant_triangles,1)*4)))
    if (axis == 1)
        imm = fb["im"][Int(plane_dist), :, :]
    end
    if (axis == 2)
        imm = fb["im"][:, Int(plane_dist), :]
    end
    if (axis == 3)
        imm = fb["im"][:, :, Int(plane_dist)]
    end
    close(fb)



    return imm, res, line_indices
end



# imm, supervoxel_vertices, supervoxel_indices = get_example_sv_to_render()

function renderSupervoxelLines(forDisplayConstants, supervoxel, mainRect)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)

    # Switch to crosshair shader and render crosshair
    glUseProgram(supervoxel.shaderProgram)
    glBindVertexArray(supervoxel.vao[])
    # glDrawElements(GL_LINES, 4, GL_UNSIGNED_INT, C_NULL)
    glDrawElements(GL_LINES, Int(round(length(supervoxel_indices) / 2)), GL_UNSIGNED_INT, C_NULL)

    # Switch back to main shader program
    # using the shader program from the mainRect causes the image render to disappear, so better use the one from forDisplayConstants !!
    glUseProgram(forDisplayConstants.shader_program)
    glBindVertexArray(mainRect.vao[])

    GLFW.SwapBuffers(forDisplayConstants.window)
end

# function updateSupervoxels(forDisplayConstants, supervoxel, mainRect, state)
#     updateImagesDisplayed(state.currentlyDispDat, state.mainForDisplayObjects, state.textDispObj, state.calcDimsStruct, state.valueForMasToSet, state.crosshairFields, state.mainRectFields, state.displayMode)

#     renderLines(forDisplayConstants, supervoxel, mainRect)
# end




# Float32[
#     0.1, 0.0, 0.0,  # top right
#     -0.1, 0.0, 0.0,  # bottom right
#     0.0, -0.1, 0.0,  # bottom left
#     0.0, 0.1, 0.0   # top left
# ]

# # Indices for drawing lines
# supervoxel_indices = UInt32[
#     0, 1,  # Line from top right to bottom right
#     2, 3   # Line from bottom left to top left
# ]


end


"""
Stuff to do :
coordinates are from -1 to 1 if text is right 20 percent of a viewer you need to transform coordinate system of verticies to be from minus 1 to 0.6 ; so first add on then divide by 2 then multiply by 1.6 (of course calculate this value) and subtract 1. The image should be the one from hdf5 just load it as medimage with spacing 1 1 1
"""
