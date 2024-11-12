using HDF5
using Statistics,LinearAlgebra
using KernelAbstractions, Test



function  get_num_tetr_in_sv()
    return 48
end


"""
given a tensor of shape (sv_i,t_i,p_i,p_c) where sv_i is index of supervoxel 
,t_i is index of tetrahedron in supervoxel ;
 p_i is index of point in tetrahedron and p_c is index of coordinate in point (so has length 3) 
we want to get a tensor of shape (sv_i,t_i*p_i,l_i,p_c) where l_i is index of line in tetrahedron and has length 2
and p_c is index of coordinate in point (so has length 3) it should store all the sections of triangles 
that are present in array, so for each triangle we will get 3 lines that are defined by 2 points in 3d space
perform it by selecting selecting diffrent points of the triangles in original array then reshape them and concatenate
. Do not use for loops 
"""
function lines_from_triangles(tensor)
    # Extract points
    p1 = tensor[:, :, 1, :]
    p2 = tensor[:, :, 2, :]
    p3 = tensor[:, :, 3, :]
    p1 = reshape(p1, size(p1, 1), size(p1, 2), 1, size(p1, 3))
    p2 = reshape(p2, size(p2, 1), size(p2, 2), 1, size(p2, 3))
    p3 = reshape(p3, size(p3, 1), size(p3, 2), 1, size(p3, 3))

    # Form lines
    l1 = cat(p1, p2, dims=3)
    l2 = cat(p2, p3, dims=3)
    l3 = cat(p3, p1, dims=3)

    # Concatenate lines
    lines = cat(l1, l2, l3, dims=3)

    # Reshape to the desired shape (sv_i, t_i * p_i, l_i, p_c)
    new_shape = (size(tensor, 1), size(tensor, 2) * 3, 2, size(tensor, 4))
    reshaped_tensor = reshape(lines, new_shape)

    return reshaped_tensor
end


function add_first_dim_index(lines_tetr_dat)
    # Get dimensions
    sv_i, t_i, p_i, _ = size(lines_tetr_dat)
    
    # Create index tensor matching original dimensions
    index_tensor = Float64.(reshape(1:sv_i, :, 1, 1, 1) .* ones(1, t_i, p_i, 1))
    
    # Concatenate along last dimension
    return cat(lines_tetr_dat, index_tensor, dims=4)
end



function get_intersection_point_augmented(tetr_dat, axis, plane_dist)
    tetr_dat=permutedims(tetr_dat, [2, 1, 3, 4, 5])
    #lets combine it into lines

    #we will work one per batch
    tetr_dat=tetr_dat[:,:,:,:,1]
    lines_tetr_dat=lines_from_triangles(tetr_dat)

    #adding index of super voxel to last dimension
    lines_tetr_dat=add_first_dim_index(lines_tetr_dat)
    #now we can get intersection points with the plane of the lines we defined above
    #in order for a triangle to intersect the plane it has to have at least one point on one side of the plane and at least one point on the other side
    bool_ind=Bool.(Bool.((lines_tetr_dat[:,:, 1, axis] .< (plane_dist)).*(lines_tetr_dat[:,:, 2, axis] .> (plane_dist))) .||
                    Bool.((lines_tetr_dat[:,:, 2, axis] .< (plane_dist)).*(lines_tetr_dat[:,:, 1, axis] .> (plane_dist))))

    #we have only lines that are relevant to the plane
    relevant_lines=Float32.(lines_tetr_dat[bool_ind,:,:])
    d1=relevant_lines[:,1,axis].-plane_dist
    d2=relevant_lines[:,2,axis].-plane_dist
    t=d1 / (d1 - d2)
    intersection_points = relevant_lines[:,1,1:3] + t * (relevant_lines[:,2,1:3]- relevant_lines[:,1,1:3])

    intersection_points_aug=cat(intersection_points,relevant_lines[:,1,4],dims=2)
    #sorted by supervoxel index
    intersection_points_aug = intersection_points_aug[sortperm(intersection_points_aug[:, end]), :]
    
    #get to opengl coordinates
    intersection_points_aug[:,1]=intersection_points_aug[:,1].-minimum(intersection_points_aug[:,1])
    intersection_points_aug[:,2]=intersection_points_aug[:,2].-minimum(intersection_points_aug[:,2])
    intersection_points_aug[:,3]=intersection_points_aug[:,3].-minimum(intersection_points_aug[:,3])
    
    intersection_points_aug[:,1]=intersection_points_aug[:,1]./maximum(intersection_points_aug[:,1])
    intersection_points_aug[:,2]=intersection_points_aug[:,2]./maximum(intersection_points_aug[:,2])
    intersection_points_aug[:,3]=intersection_points_aug[:,3]./maximum(intersection_points_aug[:,3])
    
    intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].*2
    intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].-1
        
    return intersection_points_aug
end

function create_supervoxel_dict(intersection_points_aug)
    # Extract unique supervoxel indices
    supervoxel_indices = unique(intersection_points_aug[:, 4])
    
    # Initialize the dictionary
    supervoxel_dict = Dict{Float64, Array{Float64, 2}}()
    
    # Iterate over each unique supervoxel index
    for idx in supervoxel_indices
        # Filter points that belong to the current supervoxel index
        points = intersection_points_aug[intersection_points_aug[:, 4] .== idx, 1:3]
        
        # Remove duplicate points
        unique_points = unique(points, dims=1)
        
        # Store the unique points in the dictionary
        supervoxel_dict[idx] = unique_points
    end
    
    return supervoxel_dict
end




# h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"

# axis=2
# plane_dist=41.0
# radiuss = (Float32(4.5), Float32(4.5), Float32(4.5))

# fb = h5open(h5_path_b, "r")
# #we want only external triangles so we ignore the sv center
# # we also ignore interpolated variance value here
# tetr_dat=fb["tetr_dat"][:,2:4,1:3,:]
# #we need to reshape the data to have sv index as first dimension and index of tetrahedron in sv as a second


# tetr_s=size(tetr_dat)
# batch_size=tetr_s[end]
# tetr_dat = reshape(tetr_dat, (get_num_tetr_in_sv(), Int(round(tetr_s[1] / get_num_tetr_in_sv())), tetr_s[2], tetr_s[3], batch_size))
# intersection_points_aug=get_intersection_point_augmented(tetr_dat, axis, plane_dist)
# intersection_points_aug=create_supervoxel_dict(intersection_points_aug)











# h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"

# axis=2
# plane_dist=41.0
# radiuss = (Float32(4.5), Float32(4.5), Float32(4.5))

# fb = h5open(h5_path_b, "r")
# #we want only external triangles so we ignore the sv center
# # we also ignore interpolated variance value here
# tetr_dat=fb["tetr_dat"][:,2:4,1:3,:]
# #we need to reshape the data to have sv index as first dimension and index of tetrahedron in sv as a second


# tetr_s=size(tetr_dat)
# batch_size=tetr_s[end]
# tetr_dat = reshape(tetr_dat, (get_num_tetr_in_sv(), Int(round(tetr_s[1] / get_num_tetr_in_sv())), tetr_s[2], tetr_s[3], batch_size))


# tetr_dat=permutedims(tetr_dat, [2, 1, 3, 4, 5])
# #lets combine it into lines

# #we will work one per batch
# tetr_dat=tetr_dat[:,:,:,:,1]
# lines_tetr_dat=lines_from_triangles(tetr_dat)


# tetr_dat[1,1,1,:]
# lines_tetr_dat[1,1,:,:]

# #adding index of super voxel to last dimension
# lines_tetr_dat=add_first_dim_index(lines_tetr_dat)
# #now we can get intersection points with the plane of the lines we defined above
# #in order for a triangle to intersect the plane it has to have at least one point on one side of the plane and at least one point on the other side
# bool_ind=Bool.(Bool.((lines_tetr_dat[:,:, 1, axis] .< (plane_dist)).*(lines_tetr_dat[:,:, 2, axis] .> (plane_dist))) .||
#                 Bool.((lines_tetr_dat[:,:, 2, axis] .< (plane_dist)).*(lines_tetr_dat[:,:, 1, axis] .> (plane_dist))))

# #we have only lines that are relevant to the plane
# relevant_lines=Float32.(lines_tetr_dat[bool_ind,:,:])
# d1=relevant_lines[:,1,axis].-plane_dist
# d2=relevant_lines[:,2,axis].-plane_dist
# t=d1 / (d1 - d2)
# intersection_points = relevant_lines[:,1,1:3] + t * (relevant_lines[:,2,1:3]- relevant_lines[:,1,1:3])

# intersection_points_aug=cat(intersection_points,relevant_lines[:,1,4],dims=2)
# #sorted by supervoxel index
# intersection_points_aug = intersection_points_aug[sortperm(intersection_points_aug[:, end]), :]

# #get to opengl coordinates
# intersection_points_aug[:,1]=intersection_points_aug[:,1].-minimum(intersection_points_aug[:,1])
# intersection_points_aug[:,2]=intersection_points_aug[:,2].-minimum(intersection_points_aug[:,2])
# intersection_points_aug[:,3]=intersection_points_aug[:,3].-minimum(intersection_points_aug[:,3])

# intersection_points_aug[:,1]=intersection_points_aug[:,1]./maximum(intersection_points_aug[:,1])
# intersection_points_aug[:,2]=intersection_points_aug[:,2]./maximum(intersection_points_aug[:,2])
# intersection_points_aug[:,3]=intersection_points_aug[:,3]./maximum(intersection_points_aug[:,3])

# intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].*2
# intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].-1