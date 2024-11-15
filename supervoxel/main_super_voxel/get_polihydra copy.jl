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
    lines = cat(l1, l2, l3, dims=2)

    # Reshape to the desired shape (sv_i, t_i * p_i, l_i, p_c)
    # new_shape = (size(tensor, 1), size(tensor, 2) * 3, 2, size(tensor, 4))
    # reshaped_tensor = reshape(lines, new_shape)

    return lines
end


# function add_first_dim_index(lines_tetr_dat)
#     # Get dimensions
#     sv_i, t_i, p_i, _ = size(lines_tetr_dat)
    
#     # Create index tensor matching original dimensions
#     index_tensor = Float64.(reshape(1:sv_i, :, 1, 1, 1) .* ones(1, t_i, p_i, 1))
    
#     # Concatenate along last dimension
#     return cat(lines_tetr_dat, index_tensor, dims=4)
# end



# function get_intersection_point_augmented(tetr_dat, axis, plane_dist)
#     tetr_dat=permutedims(tetr_dat, [2, 1, 3, 4, 5])
#     #lets combine it into lines

#     #we will work one per batch
#     tetr_dat=tetr_dat[:,:,:,:,1]
#     lines_tetr_dat=lines_from_triangles(tetr_dat)

#     #adding index of super voxel to last dimension
#     lines_tetr_dat=add_first_dim_index(lines_tetr_dat)
#     #now we can get intersection points with the plane of the lines we defined above
#     #in order for a triangle to intersect the plane it has to have at least one point on one side of the plane and at least one point on the other side
#     bool_ind=Bool.(Bool.((lines_tetr_dat[:,:, 1, axis] .< (plane_dist)).*(lines_tetr_dat[:,:, 2, axis] .> (plane_dist))) .||
#                     Bool.((lines_tetr_dat[:,:, 2, axis] .< (plane_dist)).*(lines_tetr_dat[:,:, 1, axis] .> (plane_dist))))

#     #we have only lines that are relevant to the plane
#     relevant_lines=Float32.(lines_tetr_dat[bool_ind,:,:])
#     d1=relevant_lines[:,1,axis].-plane_dist
#     d2=relevant_lines[:,2,axis].-plane_dist
#     t=d1 / (d1 - d2)
#     intersection_points = relevant_lines[:,1,1:3] + t * (relevant_lines[:,2,1:3]- relevant_lines[:,1,1:3])

#     intersection_points_aug=cat(intersection_points,relevant_lines[:,1,4],dims=2)
#     #sorted by supervoxel index
#     intersection_points_aug = intersection_points_aug[sortperm(intersection_points_aug[:, end]), :]
    
#     #get to opengl coordinates
#     intersection_points_aug[:,1]=intersection_points_aug[:,1].-minimum(intersection_points_aug[:,1])
#     intersection_points_aug[:,2]=intersection_points_aug[:,2].-minimum(intersection_points_aug[:,2])
#     intersection_points_aug[:,3]=intersection_points_aug[:,3].-minimum(intersection_points_aug[:,3])
    
#     intersection_points_aug[:,1]=intersection_points_aug[:,1]./maximum(intersection_points_aug[:,1])
#     intersection_points_aug[:,2]=intersection_points_aug[:,2]./maximum(intersection_points_aug[:,2])
#     intersection_points_aug[:,3]=intersection_points_aug[:,3]./maximum(intersection_points_aug[:,3])
    
#     intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].*2
#     intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].-1
        



#     return intersection_points_aug
# end

function create_supervoxel_dict(intersection_points_aug)
    # Extract unique supervoxel indices
    supervoxel_indices = unique(intersection_points_aug[:, 4])
    
    # Initialize the dictionary
    supervoxel_dict = Dict{Float64, Array{Float64, 2}}()
    
    # Iterate over each unique supervoxel index
    for idx in supervoxel_indices
        # Filter points that belong to the current supervoxel index
        points = intersection_points_aug[intersection_points_aug[:, 4] .== idx, :]
        
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




"""
Extends the last dimension of a 4D tensor from length 3 to length 6 by appending the indices from the first three dimensions.

Parameters:
- `tensor::Array{T, 4}`: A 4D tensor where the last dimension has length 3.

Returns:
- `result::Array{T, 4}`: A 4D tensor with the last dimension extended to length 6.
  After the operation, `result[i, j, k, 4] == i`, `result[i, j, k, 5] == j`, and `result[i, j, k, 6] == k`.

Example:
```julia
tetr_dat = rand(2, 2, 2, 3)
result = add_indices(tetr_dat)
println(result[1, 1, 1, 4])  # Outputs 1
println(result[1, 1, 1, 5])  # Outputs 1
println(result[1, 1, 1, 6])  # Outputs 1
```
"""
function add_indices_to_tensor(tensor)

    sz = size(tensor)
    @assert sz[4] == 3 "The last dimension of the tensor must have length 3."

    # Create index tensors
    idx1 = reshape(collect(1:sz[1]), sz[1], 1, 1, 1)
    idx2 = reshape(collect(1:sz[2]), 1, sz[2], 1, 1)
    idx3 = reshape(collect(1:sz[3]), 1, 1, sz[3], 1)

    idx1_tensor = repeat(idx1, 1, sz[2], sz[3], 1)
    idx2_tensor = repeat(idx2, sz[1], 1, sz[3], 1)
    idx3_tensor = repeat(idx3, sz[1], sz[2], 1, 1)

    # Concatenate indices with the original tensor
    idx_tensor = cat(idx1_tensor, idx2_tensor, idx3_tensor, dims=4)
    result = cat(tensor, idx_tensor, dims=4)
    return result
end

# Test case
function test_add_indices()
    # Create a sample tensor of size (2, 2, 2, 3)
    tetr_dat = rand(30, 30, 30, 3)
    result = add_indices_to_tensor(tetr_dat)

    # Check the size of the result
    # @assert size(result) == (2, 2, 2, 6)

    # Verify that the indices are correctly added
    @assert result[1, 1, 1, 4] == 1
    @assert result[7, 1, 1, 4] == 7
    @assert result[1, 21, 1, 5] == 21
    @assert result[1, 1, 27, 6] == 27
    @assert result[2, 2, 2, 4] == 2
    @assert result[2, 2, 2, 5] == 2
    @assert result[2, 2, 2, 6] == 2

    println("All tests passed.")
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
    example_sv=example_sv[:,2:4,1:4]
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
                    z = round(point[3], digits=6),
                    c = round(point[4], digits=6),
                ))
            end
        end
    points_df = DataFrame(points_list)
    
    # Step 2: Identify unique points based on coordinates
    unique_points = unique(points_df[:, [:x, :y, :z, :c]])
    
    # Step 3: Create global point IDs
    unique_points[!, :point_id] = 1:nrow(unique_points)
    
    # Step 4: Join with original dataframe to get point_ids
    points_df = leftjoin(points_df, unique_points, on=[:x, :y, :z, :c])
    points_df = filter(row -> row.x >= 0, points_df)
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
        #   print("\n triangle_indices $(triangle_indices) \n neighbor_tuples $(neighbor_tuples) \n")
          
          # Prepare the keys (multiple keys with identical value)
          key_rows = points_df[points_df.point_id .== pid, :]
          keys = [(row.triangle_index, row.point_index) for row in eachrow(key_rows)]
    
          # For each key, assign the neighbor list as value
          for key in keys
              neighbor_dict[key] = unique(collect(neighbor_tuples))
          end
      end
      return neighbor_dict
    end      



    function group_supervoxel_points(res_dict, neighbor_dict)
        grouped_points = []
        
        for (sv_index, points_matrix) in res_dict
            num_points = size(points_matrix, 1)
            points_matrix[:, 7] .= 0  # Reset group assignments
            group_index = 1
            
            while any(points_matrix[:, 7] .== 0)
                # Find first unassigned point
                start_idx = findfirst(points_matrix[:, 7] .== 0)
                current_group = Int[]
                neighbor_map = Dict{Int, Set{Int}}()
                
                # Start BFS from this point
                queue = [start_idx]
                points_matrix[start_idx, 7] = group_index
                push!(current_group, start_idx)
                
                while !isempty(queue)
                    current_idx = popfirst!(queue)
                    current_point = points_matrix[current_idx, :]
                    
                    # Get both sets of parent indices
                    parents = [
                        (current_point[5], current_point[6]),
                        (current_point[8], current_point[9])
                    ]
                    
                    # Find all neighbors through both parent points
                    for parent in parents
                        for neighbor in get(neighbor_dict, parent, [])
                            # Find points matching this neighbor's indices
                            for (idx, point) in enumerate(eachrow(points_matrix))
                                if ((point[5], point[6]) == neighbor || 
                                    (point[8], point[9]) == neighbor) && 
                                    points_matrix[idx, 7] == 0
                                    
                                    points_matrix[idx, 7] = group_index
                                    push!(queue, idx)
                                    push!(current_group, idx)
                                    
                                    # Store bidirectional neighbor relationship
                                    if !haskey(neighbor_map, current_idx)
                                        neighbor_map[current_idx] = Set{Int}()
                                    end
                                    if !haskey(neighbor_map, idx)
                                        neighbor_map[idx] = Set{Int}()
                                    end
                                    push!(neighbor_map[current_idx], idx)
                                    push!(neighbor_map[idx], current_idx)
                                end
                            end
                        end
                    end
                end
                
                # Process the current group
                if length(current_group) > 2
                    # Order points to form a cycle
                    ordered_points = order_points_cycle(current_group, neighbor_map)
                    if !isempty(ordered_points)
                        push!(grouped_points, (sv_index, points_matrix[ordered_points, :]))
                    end
                else
                    # Mark small groups as invalid
                    for idx in current_group
                        points_matrix[idx, 7] = -1
                    end
                end
                
                group_index += 1
            end
        end
        
        return grouped_points
    end
    
    function order_points_cycle(group_points, neighbor_map)
        if isempty(group_points)
            return Int[]
        end
        
        ordered = Int[group_points[1]]
        used = Set([group_points[1]])
        
        # Build path
        while length(ordered) < length(group_points)
            current = ordered[end]
            next_point = nothing
            
            # Find an unused neighbor
            for neighbor in neighbor_map[current]
                if !(neighbor in used)
                    next_point = neighbor
                    break
                end
            end
            
            if next_point === nothing
                # No unused neighbors found - check if we can close the loop
                if length(ordered) >= 3 && ordered[1] in neighbor_map[current]
                    break
                else
                    return Int[] # Cannot form valid cycle
                end
            end
            
            push!(ordered, next_point)
            push!(used, next_point)
        end
        
        # Verify the cycle
        if length(ordered) >= 3 && ordered[end] in neighbor_map[ordered[1]]
            return ordered
        end
        
        return Int[]
    end
    
    function is_group_closed_loop(points_matrix, group_points, neighbor_dict)
        if length(group_points) < 3
            return false
        end
        first_idx = group_points[1]
        last_idx = group_points[end]
        first_point = points_matrix[first_idx, :]
        last_point = points_matrix[last_idx, :]
        # Get parent indices
        first_parents = [(first_point[5], first_point[6]), (first_point[8], first_point[9])]
        last_parents = [(last_point[5], last_point[6]), (last_point[8], last_point[9])]
        # Check if first and last points are neighbors
        for fp in first_parents
            neighbors = get(neighbor_dict, fp, [])
            for lp in last_parents
                if lp in neighbors
                    return true
                end
            end
        end
        return false
    end






    
"""
    1) we need to prepare earlier dict on the basis of pure int basic tetrdat of the neighbouring indicies within supervoxel 
    2) in case of each intersection point we need to preserve information about original indexes of both points 
    3) then in a loop we are 
        a)looking on the basis of prepared earlier dict to the first point associated with a supervoxel - we mutate the points la last entry (we need to add space for it) 
        with the integer of a current group so the points will be all 0 at the begining then we select one point and set it to one - on the bases of dict from 1 
        we look for neighbours of each end of the segment that given us this intersection point and we set the integer of the group to the last entry of the point/points that is neigbour - then we are checking those newly found 
        neigbours and repeat process; during the process we are additionally reordering the points in a group so points connected should be next to each other 
        also first and last point in a list has to be connected to each other
        b)if we do not find any neigbours for a given point (we know it on the basis of the fact that last entrt remain 0) we set it to the next integer in last position of the point and
        we are checking weather any of the remaning 0's points are connected to it if they are we are setting them to the same integer etc 
        c) we are repeating the process until there are no more 0's in the last entry of the points
        d) we filter out all of the groups that has less then 3 points 
        e) we add barycenter to the group 
        f) as we know that points next to each other are neighbours we can just connect them to the barycenter and we have a triangulation what will be usefull later
    
"""
    



    h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"

    axis=3
    plane_dist=21.0
    radiuss = (Float32(4.5), Float32(4.5), Float32(4.5))

    fb = h5open(h5_path_b, "r")
    #we want only external triangles so we ignore the sv center
    # we also ignore interpolated variance value here
    tetr_dat=fb["tetr_dat"][:,2:4,1:3,:]
    #we need to reshape the data to have sv index as first dimension and index of tetrahedron in sv as a second





    





    tetr_s=size(tetr_dat)
    batch_size=tetr_s[end]
    
    ### getting information about what points are connected to what based on index in supervoxel 
    
    tetr_int=initialize_for_tetr_dat((10.0,10.0,10.0),(1.0,1.0,1.0))
    tetr_s_int=size(tetr_int)
    tetr_int = reshape(tetr_int, (get_num_tetr_in_sv(), Int(round(tetr_s_int[1] / get_num_tetr_in_sv())), tetr_s_int[2], tetr_s_int[3]))
    tetr_int=permutedims(tetr_int, [2, 1, 3, 4])
    
    sv_ind=12
    example_sv=tetr_int[sv_ind,:,:,:]
    #step 1
    neighbor_dict = create_neighbor_dict(example_sv)

    tetr_dat = reshape(tetr_dat, (get_num_tetr_in_sv(), Int(round(tetr_s[1] / get_num_tetr_in_sv())), tetr_s[2], tetr_s[3], batch_size))
    tetr_dat=permutedims(tetr_dat, [2, 1, 3, 4, 5])
    #lets combine it into lines
    #we will work one per batch
    tetr_dat=tetr_dat[:,:,:,:,1]
    tetr_dat=add_indices_to_tensor(tetr_dat)

    tetr_dat[778,12,1,:]


    lines_tetr_dat=lines_from_triangles(tetr_dat)

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
    #big step 2
    intersection_points_aug=cat(intersection_points,relevant_lines[:,1,4:6],relevant_lines[:,2,4:6],dims=2)
    #sorted by supervoxel index
    intersection_points_aug = intersection_points_aug[sortperm(intersection_points_aug[:, 4]), :]
    intersection_points_aug[:,7].=0 #setting entry 7 to 0 as it will be used for grouping

    #get to opengl coordinates
    intersection_points_aug[:,1]=intersection_points_aug[:,1].-minimum(intersection_points_aug[:,1])
    intersection_points_aug[:,2]=intersection_points_aug[:,2].-minimum(intersection_points_aug[:,2])
    intersection_points_aug[:,3]=intersection_points_aug[:,3].-minimum(intersection_points_aug[:,3])

    intersection_points_aug[:,1]=intersection_points_aug[:,1]./maximum(intersection_points_aug[:,1])
    intersection_points_aug[:,2]=intersection_points_aug[:,2]./maximum(intersection_points_aug[:,2])
    intersection_points_aug[:,3]=intersection_points_aug[:,3]./maximum(intersection_points_aug[:,3])

    intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].*2
    intersection_points_aug[:,1:3]=intersection_points_aug[:,1:3].-1
    
    res_dict=create_supervoxel_dict(intersection_points_aug)








# function group_supervoxel_points_my(sv_index, points_matrix, neighbor_dict)

tuple_dicts=collect(res_dict)


curr_index=19
sv_index, points_matrix=tuple_dicts[curr_index]


num_points = size(points_matrix, 1)
points_matrix[:, 7] .= 0  # Reset group assignments
group_index = 1

while any(points_matrix[:, 7] .== 0)
#get first point
num_points = size(points_matrix, 1)
# points_matrix[:, 7] .= 0  # Reset group assignments
group_index = 1

# while any(points_matrix[:, 7] .== 0)
    # Find first unassigned point
    start_idx = findfirst(points_matrix[:, 7] .== 0)
    current_group = Int[]
    neighbor_map = Dict{Int, Set{Int}}()
    
    # Start BFS from this point
    queue = [start_idx]
    points_matrix[start_idx, 7] = group_index
    push!(current_group, start_idx)
    
    while !isempty(queue)
        current_idx = popfirst!(queue)
        current_point = points_matrix[current_idx, :]
        
        # Get both sets of parent indices
        parents = [
            (current_point[5], current_point[6]),
            (current_point[8], current_point[9])
        ]
        
        # Find all neighbors through both parent points
        for parent in parents

            for neighbor in get(neighbor_dict, parent, [])
                # Find points matching this neighbor's indices
                for (idx, point) in enumerate(eachrow(points_matrix))
                    if ((point[5], point[6]) == neighbor || 
                        (point[8], point[9]) == neighbor) && 
                        points_matrix[idx, 7] == 0
                        
                        points_matrix[idx, 7] = group_index
                        push!(queue, idx)
                        push!(current_group, idx)
                        
                        # Store bidirectional neighbor relationship
                        if !haskey(neighbor_map, current_idx)
                            neighbor_map[current_idx] = Set{Int}()
                        end
                        if !haskey(neighbor_map, idx)
                            neighbor_map[idx] = Set{Int}()
                        end
                        push!(neighbor_map[current_idx], idx)
                        push!(neighbor_map[idx], current_idx)
                    end
                end
            end
        end
    # end
end
end

print(unique(points_matrix[:,7]))

    unique(get(neighbor_dict, (19.0,1.0), []))

    current_group
    neighbor_map

    a=1
# end



























# tetr_int=initialize_for_tetr_dat((10.0,10.0,10.0),(1.0,1.0,1.0))
# tetr_s_int=size(tetr_int)
# tetr_int = reshape(tetr_int, (get_num_tetr_in_sv(), Int(round(tetr_s_int[1] / get_num_tetr_in_sv())), tetr_s_int[2], tetr_s_int[3]))
# tetr_int=permutedims(tetr_int, [2, 1, 3, 4])

# sv_ind=12
# example_sv=tetr_int[sv_ind,:,:,:]
# example_sv=example_sv[:,2:4,1:4]
# t_i, p_i, p_c = size(example_sv)  # Dimensions of the tensor
    
# # Step 1: Flatten the tensor into a DataFrame-like structure
# points_list = []
#     for t = 1:t_i
#         for p = 1:p_i
#             point = example_sv[ t, p, :]
#             push!(points_list, (
#                 triangle_index = t,
#                 point_index = p,
#                 x = round(point[1], digits=6),  # Round coordinates to avoid floating point issues
#                 y = round(point[2], digits=6),
#                 z = round(point[3], digits=6),
#                 c = round(point[4], digits=6),
#             ))
#         end
#     end
# points_df = DataFrame(points_list)

# # Step 2: Identify unique points based on coordinates
# unique_points = unique(points_df[:, [:x, :y, :z, :c]])

# # Step 3: Create global point IDs
# unique_points[!, :point_id] = 1:nrow(unique_points)

# # Step 4: Join with original dataframe to get point_ids
# points_df = leftjoin(points_df, unique_points, on=[:x, :y, :z, :c])
# points_df = filter(row -> row.x >= 0, points_df)
#   # Initialize the dictionary
#   neighbor_dict = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()

#   # Get all unique point_ids
#   unique_point_ids = unique(points_df.point_id)

#   # Iterate over each point_id
#   for pid in unique_point_ids
#       # Get all unique triangle indices where this point occurs
#       triangle_indices = unique(points_df[points_df.point_id .== pid, :triangle_index])
#       # Initialize a set to store neighbor tuples
#       neighbor_tuples = Set{Tuple{Int, Int}}()

#       # For each triangle_index
#       for tid in triangle_indices
#           # Select all rows with this triangle_index
#           triangle_rows = points_df[points_df.triangle_index .== tid, :]

#           # Filter out rows with current point_id
#           neighbor_rows = triangle_rows[triangle_rows.point_id .!= pid, :]
#           # Extract tuples (triangle_index, point_index)
#           new_neighbors = [(row.triangle_index, row.point_index) for row in eachrow(neighbor_rows)]
#           print("\n  pid $(pid) \n neighbor_rows $(new_neighbors)\n")

#           # Add to the set of neighbor tuples
#           neighbor_tuples = union(neighbor_tuples, new_neighbors)
#       end
#     #   print("\n triangle_indices $(triangle_indices) \n neighbor_tuples $(neighbor_tuples) \n")
      
#       # Prepare the keys (multiple keys with identical value)
#       key_rows = points_df[points_df.point_id .== pid, :]
#       keys = [(row.triangle_index, row.point_index) for row in eachrow(key_rows)]

#       # For each key, assign the neighbor list as value
#       for key in keys
#           neighbor_dict[key] = unique(collect(neighbor_tuples))
#       end
#   end

# neighbor_dict

# oooo=map(el-> length(el[2]),collect(neighbor_dict))
# unique(oooo)
# b=2






# grouped_points = group_supervoxel_points(res_dict, neighbor_dict)
# ddd = Dict(grouped_points)

# iii=filter(gg-> gg[1]==425.0,grouped_points)
# iii[1][2]

# res_dict[425.0]
# ddd[425.0]

# a=1
#     # Sample res_dict and neighbor_dict
# res_dict = Dict{Int, Array{Float64, 2}}()
# neighbor_dict = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()

# # Example supervoxel data
# # Supervoxel index 1 with 4 points
# res_dict[1] = [
#     # x, y, z, sv_index, tri_idx1, pt_idx1, group_idx (empty), tri_idx2, pt_idx2
#     0.0 0.0 0.0 1 1 1 0 2 2;
#     1.0 0.0 0.0 1 2 2 0 1 1;
#     1.0 1.0 0.0 1 3 1 0 2 3;
#     0.0 1.0 0.0 1 2 3 0 3 1
# ]

# # Example neighbor relationships
# neighbor_dict[(1,1)] = [(2,2)]
# neighbor_dict[(2,2)] = [(1,1), (2,3)]
# neighbor_dict[(2,3)] = [(2,2)]
# neighbor_dict[(3,1)] = [(2,3)]

# # Call the function
# grouped_points = group_supervoxel_points(res_dict, neighbor_dict)

# # Output the result
# for (sv_index, points) in grouped_points
#     println("Supervoxel Index: ", sv_index)
#     println("Ordered Group Points:")
#     println(points)
# end