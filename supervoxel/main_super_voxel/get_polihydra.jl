using HDF5
using Statistics,LinearAlgebra
using KernelAbstractions, Test, Revise,DataFrames
includet("/media/jm/hddData/projects/MedEye3d.jl/supervoxel/main_super_voxel/initialize_sv.jl")


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


# function add_first_dim_index(lines_tetr_dat)
#     # Get dimensions
#     sv_i, t_i, p_i, _ = size(lines_tetr_dat)
    
#     # Create index tensor matching original dimensions
#     index_tensor = Float64.(reshape(1:sv_i, :, 1, 1, 1) .* ones(1, t_i, p_i, 1))
    
#     # Concatenate along last dimension
#     return cat(lines_tetr_dat, index_tensor, dims=4)
# end



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








function add_indices_to_tensor(tetr_dat)
    sv_i, t_i, p_i, p_c = size(tetr_dat)
    
    # Create index tensors
    sv_indices = reshape(repeat(collect(1:sv_i), inner=(t_i, p_i, 1)), sv_i, t_i, p_i, 1)
    t_indices = reshape(repeat(collect(1:t_i), inner=(sv_i, p_i, 1)), sv_i, t_i, p_i, 1)
    p_indices = reshape(repeat(collect(1:p_i), inner=(sv_i, t_i, 1)), sv_i, t_i, p_i, 1)
    
    # Concatenate the indices with the original tensor along the last dimension
    result = cat(tetr_dat, sv_indices, t_indices, p_indices, dims=4)
    
    return result
end

"""
    create_neighbor_dict(example_sv::Array{Float64, 4}) -> Dict{Int, Set{Int}}

Create a dictionary mapping points to their neighbors in supervoxel. Points are considered
neighbors if they share a triangle and have the same supervoxel index.

# Arguments
- `example_sv::Array{Float64, 4}`: 4D tensor where:
  - 1 dimension (`t_i`): triangle index
  - 2 dimension (`p_i`): point index in triangle
  - 3 dimension (`p_c`): point coordinates `(x, y, z)`

# Returns
- `Dict{Int, Set{Int}}`:
  - **Key**: `point_id`
  - **Value**: `Set` of `point_id` representing neighbors

# Example
"""
function create_neighbor_dict(example_sv)
    t_i, p_i, p_c = size(example_sv)  # Dimensions of the tensor
    
    # Step 1: Flatten the tensor into a DataFrame-like structure
    points_list = []
        for t = 1:t_i
            for p = 1:p_i
                point = example_sv[ t, p, :]
                push!(points_list, (
                    triangle_index = t,
                    point_index = p,
                    x = round(point[1], digits=6),  # Round coordinates to avoid floating point issues
                    y = round(point[2], digits=6),
                    z = round(point[3], digits=6)
                ))
            end
        end
    points_df = DataFrame(points_list)

    # Step 2: Identify unique points based on coordinates
    unique_points = unique(points_df[:, [:x, :y, :z]])
    
    # Step 3: Create global point IDs
    unique_points[!, :point_id] = 1:nrow(unique_points)

    # Step 4: Join with original dataframe to get point_ids
    points_df = leftjoin(points_df, unique_points, on=[:x, :y, :z])

      # Initialize the dictionary
      neighbor_dict = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()

      # Get all unique point_ids
      unique_point_ids = unique(points_df.point_id)
  
      # Iterate over each point_id
      for pid in unique_point_ids
          # Get all unique triangle indices where this point occurs
          triangle_indices = unique(points_df[points_df.point_id .== pid, :triangle_index])
  
          # Initialize a set to store neighbor tuples
          neighbor_tuples = Set{Tuple{Int, Int}}()
  
          # For each triangle_index
          for tid in triangle_indices
              # Select all rows with this triangle_index
              triangle_rows = points_df[points_df.triangle_index .== tid, :]
  
              # Filter out rows with current point_id
              neighbor_rows = triangle_rows[triangle_rows.point_id .!= pid, :]
  
              # Extract tuples (triangle_index, point_index)
              new_neighbors = [(row.triangle_index, row.point_index) for row in eachrow(neighbor_rows)]
  
              # Add to the set of neighbor tuples
              neighbor_tuples = union(neighbor_tuples, new_neighbors)
          end
  
          # Prepare the keys (multiple keys with identical value)
          key_rows = points_df[points_df.point_id .== pid, :]
          keys = [(row.triangle_index, row.point_index) for row in eachrow(key_rows)]
  
          # For each key, assign the neighbor list as value
          for key in keys
              neighbor_dict[key] = collect(neighbor_tuples)
          end
      end
  
      return neighbor_dict
    end      



example_sv = zeros(Float64, 2,3, 3)
# First supervoxel, first triangle
example_sv[1,1,:] = [0.0, 0.0, 0.0]  # Point 1
example_sv[1,2,:] = [1.0, 0.0, 0.0]  # Point 2
example_sv[1,3,:] = [0.0, 1.0, 0.0]  # Point 3

# Second supervoxel, first triangle
example_sv[2,1,:] = [4.0, 0.0, 0.0]  # Point 4
example_sv[2,2,:] = [0.0, 0.0, 0.0]  # Point 5
example_sv[2,3,:] = [2.0, 1.0, 0.0]  # Point 6


neighbor_dict = create_neighbor_dict(example_sv)

# as defined if we will have the same point it should have the same neighbors

@test neighbor_dict[1] == neighbor_dict[4]

# Test 1: Check if points have correct neighbors
@test length(neighbor_dict[1]) == 2  # Point 1 should have 2 neighbors
@test all(n in [2, 3] for n in neighbor_dict[1])

# Test 2: Points from different supervoxels with same coordinates share the same point ID
@test neighbor_dict[1] == neighbor_dict[4]  # Points with same coordinates

# Test 3: Check if dictionary has correct number of entries
@test length(neighbor_dict) == 5  # Adjusted based on unique points

# Test cases

# Points with the same coordinates should have the same neighbors
@test neighbor_dict[1] == neighbor_dict[4]

# Test 1: Check if point 1 has correct neighbors
@test length(neighbor_dict[1]) == 2  # Point 1 should have 2 neighbors
@test all(n in [2, 3] for n in neighbor_dict[1])

# Test 2: Check if point 4 (same coordinates as point 1) has the same neighbors
@test neighbor_dict[1] == neighbor_dict[4]

# Test 3: Check if points from different supervoxels are not incorrectly connected
@test !(4 in neighbor_dict[2])  # Point 2 should not be connected to point 4

# Test 4: Check if dictionary has correct number of entries
@test length(neighbor_dict) == 5  # Adjusted based on unique points


