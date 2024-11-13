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
    create_neighbor_dict(example_sv::Array{Float64, 4}) -> Dict{Tuple{Int, Int}, Set{Tuple{Int, Int}}}

Create a dictionary mapping points to their neighbors in triangles. Points are considered
neighbors if they share a triangle and have the same supervoxel index.

# Arguments
- `example_sv::Array{Float64, 4}`: 4D tensor where:
  - First dimension (sv_i): supervoxel index
  - Second dimension (t_i): triangle index
  - Third dimension: point coordinates (x,y,z)

# Returns
- Dictionary where:
  - Key: Tuple(supervoxel_index, point_id)
  - Value: Set of Tuples(supervoxel_index, point_id) representing neighbors

# Example
```julia
example_sv = zeros(Float64, 2, 2, 3, 3)  # 2 supervoxels, 2 triangles, 3 points per triangle
example_sv[1,1,:,1] .= [0.0, 1.0, 2.0]   # First triangle of first supervoxel
example_sv[1,1,:,2] .= [0.0, 1.0, 2.0]
example_sv[1,1,:,3] .= [0.0, 1.0, 2.0]
neighbor_dict = create_neighbor_dict(example_sv)
"""
function create_neighbor_dict(example_sv::Array{Float64, 4})
    sv_i, t_i, p_i, p_c = size(example_sv)  # Dimensions of the tensor
    
    # Step 1: Flatten the tensor into a DataFrame-like structure
    points_list = []

    for sv = 1:sv_i
        for t = 1:t_i
            for p = 1:p_i
                point = example_sv[sv, t, p, :]
                push!(points_list, (
                    sv_index = sv,
                    triangle_index = t,
                    point_index = p,
                    x = point[1],
                    y = point[2],
                    z = point[3]
                ))
            end
        end
    end

    points_df = DataFrame(points_list)

    # Step 2: Identify unique points based on coordinates
    unique_points = unique(points_df[:, [:x, :y, :z]])

    # Step 3: Assign unique point IDs
    unique_points.point_id = 1:size(unique_points, 1)

    # Step 4: Create point-to-ID mapping
    points_df = leftjoin(points_df, unique_points, on=[:x, :y, :z])

    # Step 5: Build neighbor relationships
    # Create a mapping from (sv_index, triangle_index) to the point_ids in each triangle
    triangles = groupby(points_df, [:sv_index, :triangle_index])
    triangle_points = combine(triangles, :point_id => x -> x)

    # Initialize neighbor dictionary
    neighbor_dict = Dict{Tuple{Int, Int}, Set{Tuple{Int, Int}}}()

    # For each triangle, associate each point with its neighbors
    for row in eachrow(triangle_points)
        sv = row.sv_index
        t = row.triangle_index
        point_ids = row.point_id

        # For each point in the triangle
        for i in 1:length(point_ids)
            pid = point_ids[i]
            # Key: (sv_index, point_id)
            key = (sv, pid)
            # Neighbors are the other points in the triangle
            neighbors = Set((sv, id) for id in point_ids if id != pid)
            # Initialize or update the neighbor set
            if haskey(neighbor_dict, key)
                neighbor_dict[key] = neighbor_dict[key] ∪ neighbors
            else
                neighbor_dict[key] = neighbors
            end
        end
    end

    return neighbor_dict
end



example_sv = zeros(Float64, 2, 1, 3, 3)
# First supervoxel, first triangle
example_sv[1,1,1,:] = [0.0, 0.0, 0.0]  # Point 1
example_sv[1,1,2,:] = [1.0, 0.0, 0.0]  # Point 2
example_sv[1,1,3,:] = [0.0, 1.0, 0.0]  # Point 3

# Second supervoxel, first triangle
example_sv[2,1,1,:] = [2.0, 0.0, 0.0]  # Point 4
example_sv[2,1,2,:] = [3.0, 0.0, 0.0]  # Point 5
example_sv[2,1,3,:] = [2.0, 1.0, 0.0]  # Point 6

neighbor_dict = create_neighbor_dict(example_sv)

# Test 1: Check if points in first supervoxel have correct neighbors
@test length(neighbor_dict[(1,1)]) == 2  # Point 1 should have 2 neighbors
@test all(n in [(1,2), (1,3)] for n in neighbor_dict[(1,1)])

# Test 2: Check if points in second supervoxel have correct neighbors
@test length(neighbor_dict[(2,4)]) == 2  # Point 4 should have 2 neighbors
@test all(n in [(2,5), (2,6)] for n in neighbor_dict[(2,4)])

# Test 3: Check if points from different supervoxels are not neighbors
@test !any(n[1] == 2 for n in neighbor_dict[(1,1)])

# Test 4: Check if dictionary has correct number of entries
@test length(neighbor_dict) == 6  # Should have entry for each point




h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"

axis=2
plane_dist=41.0
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

sv_ind=3
example_sv=tetr_int[sv_ind,:,:,:]
sv_centerr=example_sv[1,1,1:3]
sv_center_copied=permutedims(repeat(sv_centerr,inner=(1,get_num_tetr_in_sv(),3)),(2,3,1))
#get  the difference between the center and the points so all sv should evaluate to the same 
example_sv=example_sv[:,2:4,1:3]-sv_center_copied




tetr_dat = reshape(tetr_dat, (get_num_tetr_in_sv(), Int(round(tetr_s[1] / get_num_tetr_in_sv())), tetr_s[2], tetr_s[3], batch_size))
tetr_dat=permutedims(tetr_dat, [2, 1, 3, 4, 5])
#lets combine it into lines

#we will work one per batch
tetr_dat=tetr_dat[:,:,:,:,1]
tetr_dat=add_indices_to_tensor(tetr_dat)



lines_tetr_dat=lines_from_triangles(tetr_dat)

#adding index of super voxel to last dimension
lines_tetr_dat=add_first_dim_index(lines_tetr_dat)


TODO basically we need to additionaly save the index of triangle and location basically in original tetr dat 
on this we can get idea what is connected and what not as the intersection of a plane with sv may lead to creation of 
non connected polygons - for those connected we can define triangluation by just geting a center - barycenter and then connnect it to any 2 neigbouring points 
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
    

