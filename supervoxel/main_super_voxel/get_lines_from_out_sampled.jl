using HDF5
using Statistics,LinearAlgebra
using KernelAbstractions



@kernel function get_cross_section(axis_index::Int, d::Float64, triangle_arr,res_arr)
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
                base_index=(index-1)*2*3
                res_arr[base_index+1]=intersection_points[1][1]#TODO adapt to changing axes
                res_arr[base_index+2]=intersection_points[1][2]
                res_arr[base_index+3]=intersection_points[1][3]
                res_arr[base_index+4]=intersection_points[2][1]
                res_arr[base_index+5]=intersection_points[2][2]
                res_arr[base_index+6]=intersection_points[1][3]

                res_arr[base_index+axis_index]=0.0
                res_arr[base_index+axis_index*2]=0.0


            
                # end
        # end
    # end

end


function get_example_sv_to_render()
    h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"
    fb = h5open(h5_path_b, "r")
    #we want only external triangles so we ignore the sv center
    # we also ignore interpolated variance value here
    tetr_dat=fb["tetr_dat"][:,2:4,1:3,1]


    #given axis and plane we will look for the triangles that points are less then radius times 2 from the plane
    axis=3
    plane_dist=14.0
    radiuss = (Float32(3.5), Float32(3.5), Float32(3.5))

    #in order for a triangle to intersect the plane it has to have at least one point on one side of the plane and at least one point on the other side
    bool_ind=Bool.( Bool.((tetr_dat[:, 1, axis] .< (plane_dist)).*(tetr_dat[:, 2, axis] .> (plane_dist)))
    .|| Bool.((tetr_dat[:, 2, axis] .< (plane_dist)).*(tetr_dat[:, 3, axis] .> (plane_dist)))
    .|| Bool.((tetr_dat[:, 3, axis] .< (plane_dist)).*(tetr_dat[:, 1, axis] .> (plane_dist)))    
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
    relevant_triangles=Float32.(tetr_dat[bool_ind,:,:])
    # relevant_triangles=tetr_dat

    # # relevant_triangles[:,:,1]
    # relevant_triangles[50,:,:]

    # Int(round(minimum(relevant_triangles[:,:,1])))

    res = Float32.(zeros(size(relevant_triangles,1)*2*3))
    dev = get_backend(res)
    get_cross_section(dev, 128)(axis, plane_dist,relevant_triangles,res, ndrange=(size(relevant_triangles,1)))
    KernelAbstractions.synchronize(dev)

    #GETTING TO OPENGL COORDINATE system
    res=res.-minimum(res)
    res=res./maximum(res)
    res=res.*2
    res=res.-1
    

    line_indices=UInt32.(collect(0:(size(relevant_triangles,1)*4)))
    imm=fb["im"][Int(plane_dist),:,:]
    close(fb)

    return imm, res, line_indices
end



# You are Geometry and Julia programming expert, given a tensor of line sections where first dimension is line section index 
# second is point index in line index and third x,y,z coordinate of a given point
# , write an algorithm that would be able to get a point of a crossection of the set of lines with a given plane
# . The plane will be perpendicular to either x,y or z axis and will be in distance d from the center
# . hence the arguments to the functions would be "axis" (x y or z) ; 
# d (distance of the plane to the center of coordinate system) ; line_arr (tensor where first dimension is line index second point index and third is x,y,z coordinate of a given point)




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
