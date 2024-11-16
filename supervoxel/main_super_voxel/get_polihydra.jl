using HDF5
using Statistics, LinearAlgebra
using KernelAbstractions
using DataStructures, Meshes
using Revise
using Meshes
using LinearAlgebra
using GLMakie
using StatsBase, DelaunayTriangulation


function get_cross_section(axis_index::Int, d::Float64, triangle_arr, res_arr,index)
    # Determine the axis index
    # cross_section_lines = []
    # Iterate through each triangle
    # for i in 1:size(triangle_arr, 1)
    # triangle = triangle_arr[index, :, :]
    # Extract triangle points without allocations
    point_a = @view triangle_arr[index, 1, :]
    point_b = @view triangle_arr[index, 2, :]
    point_c = @view triangle_arr[index, 3, :]

    # Compute distances from the plane for each point
    dist_a = point_a[axis_index] - d
    dist_b = point_b[axis_index] - d
    dist_c = point_c[axis_index] - d


    # Initialize number of intersections
    num_intersections = 0

    # Determine indices based on axis_index
    if axis_index == 1
        ind1 = 2
        ind2 = 3
    elseif axis_index == 2
        ind1 = 1
        ind2 = 3
    else  # axis_index == 3
        ind1 = 1
        ind2 = 2
    end

    # Calculate base index for res_arr
    base_index = (index - 1) * 8  # 8 elements per triangle (2 points * 4 values each)

    # Edge from point_a to point_b
    if dist_a * dist_b < 0
        t = dist_a / (dist_a - dist_b)
        num_intersections += 1
        offset = (num_intersections - 1) * 4  # 0 for first point, 4 for second point
        res_arr[base_index + offset + 1] = point_a[ind1] + t * (point_b[ind1] - point_a[ind1])
        res_arr[base_index + offset + 2] = point_a[ind2] + t * (point_b[ind2] - point_a[ind2])
        res_arr[base_index + offset + 3] = 1.0
        res_arr[base_index + offset + 4] = point_a[4]  # Supervoxel index or other attribute
    end

    # Edge from point_b to point_c
    if dist_b * dist_c < 0
        t = dist_b / (dist_b - dist_c)
        num_intersections += 1
        offset = (num_intersections - 1) * 4
        res_arr[base_index + offset + 1] = point_b[ind1] + t * (point_c[ind1] - point_b[ind1])
        res_arr[base_index + offset + 2] = point_b[ind2] + t * (point_c[ind2] - point_b[ind2])
        res_arr[base_index + offset + 3] = 1.0
        res_arr[base_index + offset + 4] = point_b[4]
    end

    # Edge from point_c to point_a
    if dist_c * dist_a < 0
        t = dist_c / (dist_c - dist_a)
        num_intersections += 1
        offset = (num_intersections - 1) * 4
        res_arr[base_index + offset + 1] = point_c[ind1] + t * (point_a[ind1] - point_c[ind1])
        res_arr[base_index + offset + 2] = point_c[ind2] + t * (point_a[ind2] - point_c[ind2])
        res_arr[base_index + offset + 3] = 1.0
        res_arr[base_index + offset + 4] = point_c[4]
    end

    # Optional: Handle cases with fewer than two intersections
    if num_intersections < 2
        for i in (num_intersections + 1):2
            offset = (i - 1) * 4
            res_arr[base_index + offset + 1:base_index + offset + 4] .= 0.0
        end
    end


end


function get_num_tetr_in_sv()
    return 48
end


function find_connected_components(edges)
    n = size(edges, 2)
    visited = falses(n)
    components = []

    function dfs(edge_idx, component)
        push!(component, edge_idx)
        visited[edge_idx] = true
        current_edge = edges[:, edge_idx]
        current_points = [
            current_edge[1:3],
            current_edge[5:7]
        ]

        for i in 1:n
            if !visited[i]
                other_edge = edges[:, i]
                other_points = [
                    other_edge[1:3],
                    other_edge[5:7]
                ]
                if any(p -> p in current_points, other_points)
                    dfs(i, component)
                end
            end
        end
    end

    for i in 1:n
        if !visited[i]
            component = []
            dfs(i, component)
            push!(components, component)
        end
    end

    return components
end

function is_closed_loop(component, edges)
    points = Set{Tuple{Float64,Float64,Float64}}()
    for edge_idx in component
        edge = edges[:, edge_idx]
        push!(points, (edge[1], edge[2], edge[3]))
        push!(points, (edge[5], edge[6], edge[7]))
    end
    return length(points) == length(component)
end

function divide_edges(edges)
    components = find_connected_components(edges)
    subdivisions = []

    for component in components
        if is_closed_loop(component, edges)
            push!(subdivisions, component)
        else
            for edge_idx in component
                push!(subdivisions, [edge_idx])
            end
        end
    end

    return subdivisions
end
function order_boundary_points(boundary_points, tol=1e-3)
    n = length(boundary_points)
    ordered = [boundary_points[1]]
    remaining = boundary_points[2:end]
    # tol = 1e-2  # Tolerance for approximate equality

    while length(ordered) < n
        current_tuple = ordered[end]
        current_point = current_tuple[2]
        found = false
        # print("\n  current_point $(current_point) \n remaining $(remaining) \n")
        for i in 1:length(remaining)
            next_tuple = remaining[i]
            if (isapprox(current_point, next_tuple[1], atol=tol) || isapprox(current_point, next_tuple[2], atol=tol))
                push!(ordered, next_tuple)
                # print("\n  next_tuple $(next_tuple) \n")
                deleteat!(remaining, i)
                found = true
                break
            end
        end

        if !found
            current_point = current_tuple[1]
            for i in 1:length(remaining)
                next_tuple = remaining[i]
                if (isapprox(current_point, next_tuple[1], atol=tol) || isapprox(current_point, next_tuple[2], atol=tol))
                    push!(ordered, next_tuple)
                    # print("\n  next_tuple $(next_tuple) \n")
                    deleteat!(remaining, i)
                    found = true
                    break
                end
            end

            if !found
                print("Cannot find a matching tuple. Points may not be properly connected.")
                return []
            end
        end
    end
    ordered = unique(ordered)
    #orienting edge correctly if needed
    for i in (1:(length(ordered)-1))
        curr_tuple = ordered[i]
        next_tuple = ordered[i+1]
        pred = !isapprox(curr_tuple[2], next_tuple[1], atol=tol)
        if pred
            # print("\n  curr_tuple : $curr_tuple  next_tuple $next_tuple \n")
            ordered[i+1] = [next_tuple[2], next_tuple[1]]
        end
    end
    #in order to avoid issues with floatting point erros we are mutating elements
    for i in (1:(length(ordered)-1))
        ordered[i][2] = ordered[i+1][1]
    end
    ordered[end][2] = ordered[1][1]


    # Verify that the last point connects to the first point
    # if !isapprox(ordered[end][2], ordered[1][1], atol=tol)
    #     error("The last point does not connect back to the first point.")
    # end

    return ordered
end


function get_trianglesss(points, boundary_nodes, reverse_order, just_edges, nust_points)
    res = []
    try
        if (reverse_order)
            points_reversed = reverse(points)
            tri = DelaunayTriangulation.triangulate(points_reversed)
            res = map(inds -> [get_point(tri, inds[1]), get_point(tri, inds[2]), get_point(tri, inds[3])], collect(get_triangles(tri)))
        elseif (just_edges)
            tri = DelaunayTriangulation.triangulate(points)
            segments = boundary_nodes
            res = map(inds -> [get_point(tri, inds[1]), get_point(tri, inds[2]), get_point(tri, inds[3])], collect(get_triangles(tri)))
        elseif (nust_points)
            tri = DelaunayTriangulation.triangulate(points)
            res = map(inds -> [get_point(tri, inds[1]), get_point(tri, inds[2]), get_point(tri, inds[3])], collect(get_triangles(tri)))
        else
            tri = DelaunayTriangulation.triangulate(points)
            res = map(inds -> [get_point(tri, inds[1]), get_point(tri, inds[2]), get_point(tri, inds[3])], collect(get_triangles(tri)))
        end
        #mapping from triangulation to triangle 2d points
    catch e
        # print("\n e $(e)\n")
        res = []
    end

    return res
end



function transform_edges_to_indices(edges, points)
    # Step 1: Create dictionary mapping points to indices
    point_to_index = Dict{Tuple{Float64,Float64},Int}()
    point_indicies = []
    for (i, point) in enumerate(points)
        point_to_index[round(point[1];digits=3), round(point[2];digits=3)] = i
        push!(point_indicies, i)
    end

    # Step 2: Transform edges using the dictionary
    transformed_edges =[]
    for edge in edges
        # Convert each point in edge to its index
        transformed_edge = (
            point_to_index[(round(edge[1][1];digits=3), round(edge[1][2];digits=3))],
            point_to_index[(round(edge[2][1];digits=3), round(edge[2][2];digits=3))]
        )
        push!(transformed_edges, transformed_edge)
    end

    return transformed_edges, point_indicies
end



# Example usage:
# edges = [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]]
# points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
# transformed = transform_edges_to_indices(edges, points)


# function get_example_sv_to_render()

function main_get_poligon_data(tetr_dat, axis, plane_dist, radiuss)

    tetr_s = size(tetr_dat)
    tetr_dat_max = maximum(tetr_dat)
    tetr_dat_min = minimum(tetr_dat)



    ### adding supervoxel index to each point
    tetr_dat_3d = reshape(tetr_dat, (get_num_tetr_in_sv(), Int(round(tetr_s[1] / get_num_tetr_in_sv())), tetr_s[2], tetr_s[3]))
    tetr_dat_3d = permutedims(tetr_dat_3d, [2, 1, 3, 4])
    sz = size(tetr_dat_3d)
    idx1 = reshape(collect(1:sz[1]), sz[1], 1, 1, 1)
    idx1_tensor = repeat(idx1, 1, sz[2], sz[3], 1)
    tetr_dat_3d = cat(tetr_dat_3d, idx1_tensor, dims=4)
    # get back to flattened
    tetr_dat_3d = permutedims(tetr_dat_3d, [2, 1, 3, 4])
    tetr_dat_3d = reshape(tetr_dat_3d, (tetr_s[1], tetr_s[2], tetr_s[3] + 1))
    @assert tetr_dat_3d[:, :, 1:3] == tetr_dat # test
    tetr_dat = tetr_dat_3d

    #given axis and plane we will look for the triangles that points are less then radius times 2 from the plane

    #in order for a triangle to intersect the plane it has to have at least one point on one side of the plane and at least one point on the other side
    bool_ind = Bool.(Bool.((tetr_dat[:, 1, axis] .< (plane_dist)) .* (tetr_dat[:, 2, axis] .> (plane_dist)))
                     .|| Bool.((tetr_dat[:, 2, axis] .< (plane_dist)) .* (tetr_dat[:, 3, axis] .> (plane_dist)))
                     .|| Bool.((tetr_dat[:, 3, axis] .< (plane_dist)) .* (tetr_dat[:, 1, axis] .> (plane_dist)))
    )

    # #we will only consider the triangles that intersect the plane
    relevant_triangles = Float32.(tetr_dat[bool_ind, :, :])





    res = Float32.(zeros(size(relevant_triangles, 1) * 2 * 4))
    # dev = get_backend(res)
    # get_cross_section(dev, 128)(axis, plane_dist, relevant_triangles, res, ndrange=(size(relevant_triangles, 1)))
    # KernelAbstractions.synchronize(dev)

    #non parallelized version
    for i in 1:size(relevant_triangles, 1)
        get_cross_section(axis, plane_dist, relevant_triangles, res, i)
    end    
    res_shape = size(res)
    secc = Int(round(res_shape[1] / 8))
    res_reshaped = reshape(res, (8, secc))
    @assert res_reshaped[4, :] == res_reshaped[8, :]
    #order by sv index
    sorted_indices = sortperm(res_reshaped[8, :])
    res_reshaped = res_reshaped[:, sorted_indices]
    sv_indicies = unique(res_reshaped[8, :])

    #maximum number of edges per usupervoxel multiplied to be safe 
    max_count = maximum(values(countmap(res_reshaped[8, :]))) * 3

    #TODO get preallocated triangulation result


    # curr_sv_index=sv_indicies[1]
    curr_sv_index = sv_indicies[1]
    for curr_sv_index in sv_indicies
        filtered_res = res_reshaped[:, res_reshaped[8, :].==curr_sv_index]
        subdivisions = divide_edges(filtered_res)

        # Print the result
        for (i, subdivision) in enumerate(subdivisions)
            # println(" index $curr_sv_index Subdivision $i: ", subdivision)
        end
    end

    # curr_sv_index=430.0

    curr_sv_index = 402.0

    numm = []
    all_res = []

    #get open gl coordinates
    res_reshaped[1:3, :] = res_reshaped[1:3, :] .- tetr_dat_min
    res_reshaped[1:3, :] = res_reshaped[1:3, :] ./ tetr_dat_max

    res_reshaped[1:3, :] = res_reshaped[1:3, :] .* 2
    res_reshaped[1:3, :] = res_reshaped[1:3, :] .- 1

    res_reshaped[5:7, :] = res_reshaped[5:7, :] .- tetr_dat_min
    res_reshaped[5:7, :] = res_reshaped[5:7, :] ./ tetr_dat_max

    res_reshaped[5:7, :] = res_reshaped[5:7, :] .* 2
    res_reshaped[5:7, :] = res_reshaped[5:7, :] .- 1

    for curr_sv_index in sv_indicies
        filtered_res = res_reshaped[:, res_reshaped[8, :].==curr_sv_index]
        subdivisions = divide_edges(filtered_res)
        # curr_sub_division=subdivisions[1]
        for curr_sub_division in subdivisions
            if (length(curr_sub_division) > 2)

                # curr_sub_division=subdivisions[3]
                # segms=map(i->Meshes.Segment(Meshes.Point(filtered_res[1:3, i]...), Meshes.Point(filtered_res[5:7, i]...)), 1:size(filtered_res, 2))
                # segms = []
                boundary_points = []

                # Map the segments
                for i in curr_sub_division
                    # segment = Meshes.Segment(Meshes.Point(filtered_res[1:3, i]...), Meshes.Point(filtered_res[5:7, i]...))
                    push!(boundary_points, [filtered_res[1:2, i], filtered_res[5:6, i]])
                    # push!(segms, segment)
                end
                #in case of floating point number error issues
                ordered = order_boundary_points(boundary_points)

                points_a = [filtered_res[1:2, i] for i in 1:size(filtered_res, 2)]
                points_b = [filtered_res[5:6, i] for i in 1:size(filtered_res, 2)]
                points = vcat(points_a, points_b)
                points_prim=unique(map(tt->(tt[1],tt[2]),points))
                


                unordered_edges_ind, point_indicies = transform_edges_to_indices(boundary_points, points_prim)
                unordered_edges_ind = Set([unordered_edges_ind...])
                unordered_edges_ind= filter(x->x[1]!=x[2], unordered_edges_ind)

                if (length(ordered) == 0)
                    ordered = order_boundary_points(boundary_points, 1e-2)
                end
                if (length(ordered) == 0)
                    ordered = order_boundary_points(boundary_points, 1e-1)
                end
                if (length(ordered) > 0)

                    # points_a = [filtered_res[1:2, i] for i in 1:size(filtered_res, 2)]
                    # points_b = [filtered_res[5:6, i] for i in 1:size(filtered_res, 2)]
                    # points = unique(vcat(points_a, points_b))

                    #here we get boundry constrained triangulation
                    boundary_nodes, points = DelaunayTriangulation.convert_boundary_points_to_indices(ordered)



                    res = get_trianglesss(points, boundary_nodes, false, false, false)
                    # print("### $res ###")
                    if (length(res) == 0)
                        res = get_trianglesss(points, boundary_nodes, true, false, false)
                    end
                    if (length(res) == 0)
                        res = get_trianglesss(points_prim, unordered_edges_ind, false, true, false)
                    end
                    if (length(res) == 0)
                        res = get_trianglesss(points, boundary_points, false, true, false)
                    end
                    if (length(res) == 0)
                        res = get_trianglesss(points, boundary_nodes, false, false, true)
                    end
                    if (length(res) == 0)

                        tri = DelaunayTriangulation.triangulate(points_prim; segments=unordered_edges_ind,check_arguments=false)
                        res = map(inds -> [get_point(tri, inds[1]), get_point(tri, inds[2]), get_point(tri, inds[3])], collect(get_triangles(tri)))

                    end
                    if (length(res) == 0)
                        print("  **** **** ")
                    else
                        res = map(el_list -> map(el -> (el[1], el[2], 0.0), el_list), res)
                        push!(all_res, (curr_sv_index,res))

                    end
                else
                    # if(length(points_prim)>2)

                    #     tri = DelaunayTriangulation.triangulate(points_prim; segments=unordered_edges_ind,check_arguments=false)
                    #     res = map(inds -> [get_point(tri, inds[1]), get_point(tri, inds[2]), get_point(tri, inds[3])], collect(get_triangles(tri)))
                    #     push!(all_res, (curr_sv_index,res))
                    # end

                end



            end
        end
    end
    # Returns
    return all_res
end


function get_sv_mean(out_sampled_points)
    sizz_out = size(out_sampled_points)#(65856, 9, 5)
    batch_size = size(out_sampled_points)[end]#(65856, 9, 5)
    num_sv = Int(round(sizz_out[1] / get_num_tetr_in_sv()))
    
    out_sampled_points = reshape(out_sampled_points[:, :, 1:2, :], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 2, batch_size))
    out_sampled_points = permutedims(out_sampled_points, [2, 1, 3, 4, 5])
    values_big = reshape(out_sampled_points[:, :, :, 1, :], num_sv, get_num_tetr_in_sv() * 9, batch_size)
    weights_big = reshape(out_sampled_points[:, :, :, 2, :], num_sv, get_num_tetr_in_sv() * 9, batch_size)
    
    big_weighted_values = values_big .* weights_big
    big_weighted_values_summed = sum(big_weighted_values, dims=2)
    
    big_weights_sum = sum(weights_big, dims=2)
    
    big_mean_weighted = big_weighted_values_summed ./ big_weights_sum
    return big_mean_weighted

end   


# h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"
# fb = h5open(h5_path_b, "r")
# #we want only external triangles so we ignore the sv center
# # we also ignore interpolated variance value here
# tetr_dat = fb["tetr_dat"][:, 2:4, 1:3, 1]

# axis = 3
# plane_dist = 22.0
# radiuss = (Float32(4.5), Float32(4.5), Float32(4.5))
# main_get_poligon_data(tetr_dat, axis, plane_dist, radiuss)
# all_res[23][1]

# typeof(all_res)
# TODO we can make a small simulation to get max triangles per sv multiply max number of points associated by sume sv and multiply it by 3 and use it for 
#     preallocation - we are on cpu so we can do it and then just get boolean to get used spots 

# Meshes.viz(segms)

#     fig, ax, sc = triplot(tri, show_constrained_edges = true, constrained_edge_linewidth = 6)
#     # lines!(ax, section_1, color = :red, linewidth = 6)
#     # lines!(ax, section_2, color = :green, linewidth = 6)
#     # lines!(ax, section_3, color = :blue, linewidth = 6)
#     fig

# transpose(filtered_res)
#     a=1

#     #GETTING TO OPENGL COORDINATE system
#     res=res.-minimum(res)
#     res=res./maximum(res)
#     res=res.*2
#     res=res.-1


#     line_indices=UInt32.(collect(0:(size(relevant_triangles,1)*16)))
#     # line_indices=UInt32.(collect(0:(size(relevant_triangles,1)*4)))
#     if(axis==1)
#         imm=fb["im"][Int(plane_dist),:,:]
#     end
#     if(axis==2)
#         imm=fb["im"][:,Int(plane_dist),:]
#     end
#     if(axis==3)
#         imm=fb["im"][:,:,Int(plane_dist)]
#     end
#     close(fb)

#     return imm, res, line_indices
# end


