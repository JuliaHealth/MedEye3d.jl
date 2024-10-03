using HDF5





function get_cross_section(axis_index::Int, d::Float64, triangle_arr)
    # Determine the axis index


    cross_section_lines = []

    # Iterate through each triangle
    for i in 1:size(triangle_arr, 1)
        triangle = triangle_arr[i, :, :]
        points = [triangle[j, :] for j in 1:3]

        # Check if the triangle intersects the plane
        distances = [point[axis_index] - d for point in points]
        signs = sign.(distances)

        if length(unique(signs)) == 1
            # All points are on the same side of the plane, no intersection
            continue
        end

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

        if length(intersection_points) == 2
            push!(cross_section_lines, intersection_points)
        end
    end

    return cross_section_lines
end


# function get_example_sv_to_render()
    h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"
    fb = h5open(h5_path_b, "r")
    #we want only external triangles so we ignore the sv center
    # we also ignore interpolated variance value here
    tetr_dat=fb["tetr_dat"][:,2:4,1:3,1]


    #given axis and plane we will look for the triangles that points are less then radius times 2 from the plane
    axis=1
    plane_dist=10.0
    radiuss = (Float32(3.5), Float32(3.5), Float32(3.5))

    # bool_ind=(tetr_dat[:, :, axis] .< (plane_dist + radiuss[axis])).*(tetr_dat[:, :, axis] .> (plane_dist - radiuss[axis]))
    # bool_ind=(tetr_dat[:, :, axis] .< (plane_dist+ radiuss[axis] ))#.*(tetr_dat[:, :, axis] .> (plane_dist - radiuss[axis]))
    
    bool_ind=Bool.( Bool.((tetr_dat[:, 1, axis] .< (plane_dist)).*(tetr_dat[:, 2, axis] .> (plane_dist)))
    .|| Bool.((tetr_dat[:, 2, axis] .< (plane_dist)).*(tetr_dat[:, 3, axis] .> (plane_dist)))
    .|| Bool.((tetr_dat[:, 3, axis] .< (plane_dist)).*(tetr_dat[:, 1, axis] .> (plane_dist)))    
    )

    relevant_triangles=tetr_dat[bool_ind,:,:]

    minimum(relevant_triangles[:,:,1])
    maximum(relevant_triangles[:,:,1])

    line_coords=get_cross_section(axis, plane_dist, relevant_triangles)
    line_coords=reduce(vcat,line_coords)
    line_coords2d=map(el->[el[2],el[3],0.0],line_coords)
    line_coords2d=vcat(line_coords2d...)
    line_coords2d=Float32.(line_coords2d)
    line_indices=UInt32.(collect(1:length(line_coords2d)))
    imm=fb["im"][Int(plane_dist),:,:]
    close(fb)

    # return imm, line_coords2d, line_indices
# end



# You are Geometry and Julia programming expert, given a tensor of triangles where first dimension is triangle index second is point index in triangle and third x,y,z coordinate of a given point
# , write an algorithm that would be able to get a crossection of the set of triangle with a given plane
# . The plane will be perpendicular to either x,y or z axis and will be in distance d from the center
# . hence the arguments to the functions would be "axis" (x y or z) ; 
# d (distance of the plane to the center of coordinate system) ; triangle_arr (tensor where first dimension is triangle index second point index and third is x,y,z coordinate of a given point)




# relevant_triangles





# """
# Function to filter triangles based on the given axis and plane
# tetr_data_entry is 3x3 matrix where first dimension indicates point of triangle 
# and second dimension indicates x,y,z coordinate of this point
# """
# function filter_triangles(tetr_data_entry, axis, plane_dist,radiuss)
    

# end

# # Perform threaded map operation
# filtered_triangles = Vector{Any}(undef, size(tetr_dat_reshaped, 1))
# Threads.@threads for i in 1:size(tetr_dat_reshaped, 1)
#     filtered_triangles[i] = filter_triangles(tetr_dat_reshaped[i, :, :], axis, plane)
# end
