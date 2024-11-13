
using SplitApplyCombine,KernelAbstractions

"""
get 4 dimensional array of cartesian indicies of a 3 dimensional array
thats size is passed as an argument dims
"""
function get_base_indicies_arr(dims)    
    indices = CartesianIndices(dims)
    # indices=collect.(Tuple.(collect(indices)))
    indices=Tuple.(collect(indices))
    indices=collect(Iterators.flatten(indices))
    indices=reshape(indices,(3,dims[1],dims[2],dims[3]))
    indices=permutedims(indices,(2,3,4,1))
    return indices
end#get_base_indicies_arr


function get_corrected_dim(ax,radius,image_shape)
    return Int(floor((image_shape[ax])/(radius[ax]*2)))
end    

function get_dif(ax,image_shape,dims,radius,pad)
    # return max(floor((image_shape[ax]-((dims[ax]+1).*diam))/2),2.0)+pad
    # return floor((image_shape[ax]-((dims[ax]).*(radius[ax]))))+((radius[ax]))
    return floor((image_shape[ax]-((dims[ax]).*(radius[ax]*2)))/2)+(radius[ax])

end

"""
initialize sv centers coordinates  we need sv centers that is in each axis 1 bigger than control points
"""
function get_sv_centers(radius,image_shape,pad=0.0)
    dims=(get_corrected_dim(1,radius,image_shape),get_corrected_dim(2,radius,image_shape),get_corrected_dim(3,radius,image_shape))
    diffs= (get_dif(1,image_shape,dims,radius,pad),get_dif(2,image_shape,dims,radius,pad),get_dif(3,image_shape,dims,radius,pad))

    #print("\n dddd dims $(dims) diffs $(diffs) radius $(radius);  $(radius[1])\n")

    res= (get_base_indicies_arr(dims).-1)#*diam
    res=Float32.(res)

    res[:,:,:,1]=(res[:,:,:,1].*(radius[1]*2)).+diffs[1]
    res[:,:,:,2]=(res[:,:,:,2].*(radius[2]*2)).+diffs[2]
    res[:,:,:,3]=(res[:,:,:,3].*(radius[3]*2)).+diffs[3]
    return res,dims,diffs

end


"""
flips the value of the index of the tuple at the position ind needed for get_linear_between function
"""
function flip_num(base_ind,tupl,ind)
    arr=collect(tupl)
    # arr=append!(arr,[4])
    if(arr[ind]==base_ind[ind])
        arr[ind]=base_ind[ind]+1
    else
        arr[ind]=base_ind[ind]
    end    
    return arr
end

"""
we can identify the line between two corners that go obliquely through the wall of the cube
it connects points that has 2 coordinates diffrent and one the same 
we can also find a point in the middle so it will be in lin_x if this common index is 1 and in lin_y if it is 2 and lin_z if 3
next if we have 1 it is pre and if 2 post
    control_points first dimension is lin_x, lin_y, lin_z, oblique
"""
function get_linear_between(base_ind,ind_1,ind_2)
    if(ind_1[1]==ind_2[1])
        return [ind_1[1],base_ind[2],base_ind[3],1]
    end
    if(ind_1[2]==ind_2[2])
        return [base_ind[1],ind_1[2],base_ind[3],2]
    end

    return [base_ind[1],base_ind[2],ind_1[3],3]
end


"""
helper function to set values to the all_surf_triangles array in the appropriate index
"""
function set_to_index(all_surf_triangles,add_ind, res_main_ind, el1,el2,el3,el4,el5)
    all_surf_triangles[res_main_ind+add_ind,1,:]=el1
    all_surf_triangles[res_main_ind+add_ind,2,:]=el2
    all_surf_triangles[res_main_ind+add_ind,3,:]=el3
    all_surf_triangles[res_main_ind+add_ind,4,:]=el4
    all_surf_triangles[res_main_ind+add_ind,5,:]=el5

end


"""
get a flattened array of all surface triangles of all supervoxels
in first dimension every get_num_tetr_in_sv() elements are a single supervoxel
second dimension is size 5 and is in order sv_center, point a,point b,point c,centroid of the base
    where centroid is a placeholder for centroid of the triangle a,b,c
in last dimension we have x,y,z coordinates of the point
currently we have just indicies to the appropriate arrays -> it need to be populated after weights get applied        
"""
function get_tetr_triangles_in_corner_on_kern(indices,corner_add,all_surf_triangles,index,corn_num)
    base_ind=indices[index[1],:]
    corner=(base_ind[1]+corner_add[1],base_ind[2]+corner_add[2],base_ind[3]+corner_add[3])
    
    corner=Float32.(append!(collect(corner),[4]))
    
    sv_center=Float32.([base_ind[1],base_ind[2],base_ind[3],-1.0])
    p_a=Float32.(flip_num(base_ind,corner,1))
    p_b=Float32.(flip_num(base_ind,corner,2))
    p_c=Float32.(flip_num(base_ind,corner,3))
    
    p_ab=Float32.(get_linear_between(base_ind,p_a,p_b))
    p_ac=Float32.(get_linear_between(base_ind,p_a,p_c))
    p_bc=Float32.(get_linear_between(base_ind,p_b,p_c))
    #now we know that corner is the primary oblique point - we now want to now take into account other 3 obliques
    #we also know that in each tetrahedron we have some additional corner either p_a or p_b or p_c 
    # in case of p_a it is next in x axis in p_b next in y axis and in p_c next in z axis


    oblique_x_1=copy(corner)
    oblique_x_1[4]=5
    oblique_x_1[1]=oblique_x_1[1]-corner_add[1]
    oblique_y_1=copy(corner)
    oblique_y_1[4]=6
    oblique_y_1[2]=oblique_y_1[2]-corner_add[2]
    oblique_z=copy(corner)
    oblique_z[3]=oblique_z[3]-corner_add[3]
    oblique_z[4]=7


    dummy=Float32.([-1.0,-1.0,-1.0,-1.0])
    # res_main_ind= (index[1]-1)*get_num_tetr_in_sv()+(index[2]-1)*6
    res_main_ind= (index[1]-1)*48+(index[2]-1)*12
    #moving by one as svv centers indicies are moved by 1
    sv_center=(sv_center.+1)
   
    set_to_index(all_surf_triangles,1, res_main_ind, sv_center,corner,oblique_x_1,p_ab,dummy)
    set_to_index(all_surf_triangles,2, res_main_ind, sv_center,oblique_x_1,p_a,p_ab,dummy)
    
    set_to_index(all_surf_triangles,3, res_main_ind, sv_center,oblique_y_1,p_ab,p_b,dummy)#
    set_to_index(all_surf_triangles,4, res_main_ind, sv_center,corner,p_ab,oblique_y_1,dummy)#

    set_to_index(all_surf_triangles,5, res_main_ind, sv_center,oblique_y_1,p_b,p_bc,dummy)#
    set_to_index(all_surf_triangles,6, res_main_ind, sv_center,corner,oblique_y_1,p_bc,dummy)#

    set_to_index(all_surf_triangles,7, res_main_ind, sv_center,oblique_z,p_bc,p_c,dummy)
    set_to_index(all_surf_triangles,8, res_main_ind, sv_center,corner,p_bc,oblique_z,dummy)

    set_to_index(all_surf_triangles,9, res_main_ind,sv_center,oblique_x_1,p_a,p_ac,dummy)
    set_to_index(all_surf_triangles,10, res_main_ind,sv_center,corner,oblique_x_1,p_ac,dummy)

    set_to_index(all_surf_triangles,11, res_main_ind,sv_center,oblique_z,p_ac,p_c,dummy)
    set_to_index(all_surf_triangles,12, res_main_ind,sv_center,corner,p_ac,oblique_z,dummy)

end


@kernel function set_triangles_kern(@Const(indices),all_surf_triangles)

    # index = @index(Global)
    index = @index(Global, Cartesian)
    # get_tetr_triangles_in_corner(base_ind,(base_ind[1],base_ind[2],base_ind[3]))
    if(index[2]==1)
        get_tetr_triangles_in_corner_on_kern(indices,(0.0,0.0,0.0),all_surf_triangles,index,1)
    end        
    #get_tetr_triangles_in_corner(base_ind,(base_ind[1]+1,base_ind[2]+1,base_ind[3]))
    if(index[2]==2)
        get_tetr_triangles_in_corner_on_kern(indices,(1.0,1.0,0.0),all_surf_triangles,index,2)
    end        
    #get_tetr_triangles_in_corner(base_ind,(base_ind[1],base_ind[2]+1,base_ind[3]+1))
    if(index[2]==3)
        get_tetr_triangles_in_corner_on_kern(indices,(0.0,1.0,1.0),all_surf_triangles,index,3)
    end        
    #get_tetr_triangles_in_corner(base_ind,(base_ind[1]+1,base_ind[2],base_ind[3]+1))
    if(index[2]==4)
        get_tetr_triangles_in_corner_on_kern(indices,(1.0,0.0,1.0),all_surf_triangles,index,4)
    end        

end

"""
return number of tetrahedrons in single supervoxel
"""
function get_num_tetr_in_sv()
    if(is_point_per_triangle)
        return 48*3
    end 
    return 48
end

"""
calculate shape of the tetr_dat array - array with tetrahedrons that are created by the center of the supervoxel
"""
function get_tetr_dat_shape(dims)
    # dims=(get_corrected_dim(1,radius,image_shape),get_corrected_dim(2,radius,image_shape),get_corrected_dim(3,radius,image_shape))
    # dims=dims.-1
    return (dims[1]*dims[2]*dims[3]*48,5,4)
end    

"""
get a flattened array of all surface triangles of all supervoxels
in first dimension every get_num_tetr_in_sv() elements are a single supervoxel
second dimension is size 5 and is in orde sv_center, point a,point b,point c,centroid 
    where centroid is a placeholder for centroid of the triangle a,b,c
in last dimension we have x,y,z coordinates of the point
currently we have just indicies to the appropriate arrays -> it need to be populated after weights get applied        
"""
function get_flattened_triangle_data(dims)
    dims=dims.-2
    indices = CartesianIndices(dims)
    # indices=collect.(Tuple.(collect(indices)))
    indices=Tuple.(collect(indices))
    indices=collect(Iterators.flatten(indices))
    indices=reshape(indices,(3,dims[1]*dims[2]*dims[3]))
    indices=permutedims(indices,(2,1))

    # indices=splitdims(indices,1)
    all_surf_triangles=zeros(Float32,get_tetr_dat_shape(dims))

    dev = get_backend(all_surf_triangles)
    set_triangles_kern(dev, 19)(Float32.(indices),all_surf_triangles
    , ndrange=((dims[1]*dims[2]*dims[3]),4))
    KernelAbstractions.synchronize(dev)

    # all_surf_triangles=map(el->get_all_surface_triangles_of_sv(el),indices)
    # #concatenate all on first dimension
    # all_surf_triangles=map(el->vcat(el...),all_surf_triangles)

    # all_surf_triangles=vcat(all_surf_triangles...)

    # print("\n ooo 111  $(size(all_surf_triangles)) \n")

    return all_surf_triangles
end

"""
initializing control points - to be modified later based on learnable weights
"""
function initialize_control_points(image_shape,radius,is_points_per_triangle=false)
    pad=0.0
    dims=(get_corrected_dim(1,radius,image_shape),get_corrected_dim(2,radius,image_shape),get_corrected_dim(3,radius,image_shape))
    diffs= (get_dif(1,image_shape,dims,radius,pad),get_dif(2,image_shape,dims,radius,pad),get_dif(3,image_shape,dims,radius,pad))
    #indicies_control_points
    icp=get_base_indicies_arr(dims.-1)

                        # lin_x, lin_y, lin_z, oblique_main,oblique_x,oblique_y,oblique_z
    if(is_points_per_triangle)
        res= combinedims(map(a->copy(icp), range(1,7+24)),4)
    else    
        res= combinedims([copy(icp), copy(icp), copy(icp), copy(icp),copy(icp),copy(icp),copy(icp)],4)
    end
    return res
end#initialize_centeris_and_control_points    

function initialize_for_tetr_dat(image_shape,radius,pad=0)
    sv_centers,dims,diffs= get_sv_centers(radius,image_shape,pad)
    return get_flattened_triangle_data(dims)  

end#initialize_centeris_and_control_points 



function count_zeros(arr, name::String)
    num_zeros = count(x -> x == 0.0, arr)
    num_entries = length(arr)
    percentt=(num_zeros/num_entries)*100
    println("percent of zeros in $name: $percentt % sum: $(sum(arr))  ")
end


################
#initialize for point per triangle unrolled

############



"""
given 3 by 3 matrix where first dimension is the point index and second is the 
coordinate index we will check weather the triangles are the same - so the a set of points in both cases are the same the order of 
triangle verticies in botsh cases is not important
"""
function is_equal_point(point,points)
    for i in 1:3
        
        a = point[1] ≈ points[i][1]
        b = point[2] ≈ points[i][2]
        c = point[3] ≈ points[i][3]
        d = point[4] ≈ points[i][4]
        if a && b && c && d
            return true
        end
    end
    return false
end




"""
looks weather given points are present in both arrays - order of points do not matter
"""
function are_triangles_equal(triangle1, triangle2)
    # Extract points from the matrices
    points1 = [triangle1[i, :] for i in 1:3]
    points2 = [triangle2[i, :] for i in 1:3]
    a=is_equal_point(points1[1],points2)
    b=is_equal_point(points1[2],points2)
    c=is_equal_point(points1[3],points2)
    # if(a || b || c)
    #     print("** $a $b $c **")
    # end
    return (a && b && c)
end




function check_triangles(middle_coords, tetr_3d, middle_tetr,full_data=false)
    # Generate all possible (xd, yd, zd) combinations
    coords = [(xd, yd, zd) for xd in (middle_coords[1]-1):(middle_coords[1]+1),
                              yd in (middle_coords[2]-1):(middle_coords[2]+1),
                              zd in (middle_coords[3]-1):(middle_coords[3]+1)
              if !(xd == middle_coords[1] && yd == middle_coords[2] && zd == middle_coords[3])]

    # Function to check triangles and return results
    function check_and_collect(xd, yd, zd)
        results = []
        for i in 1:48
            for j in 1:48

                rel=(xd - middle_coords[1],yd - middle_coords[2],zd - middle_coords[3])
                submatrix_tetr_3d = tetr_3d[xd, yd, zd, i, 2:4, 1:4]
                submatrix_middle_tetr = middle_tetr[j, :, :]
                are_equal = are_triangles_equal(submatrix_tetr_3d, submatrix_middle_tetr)

                triangle_points_middle=[submatrix_middle_tetr[ii, :] for ii in 1:3]
                apex1_middle=tetr_3d[middle_coords[1], middle_coords[2], middle_coords[3], j, 1, 1:4]
                apex2_middle=tetr_3d[xd, yd, zd, j, 1, 1:4]
                triangle_points=map(tp-> [(tp[1]-middle_coords[1]),(tp[2]-middle_coords[2]),(tp[3]-middle_coords[3]),tp[4]],triangle_points_middle)
                apex1=[(apex1_middle[1]- middle_coords[1]),(apex1_middle[2]- middle_coords[2]),(apex1_middle[3]- middle_coords[3])]
                apex2=[(apex2_middle[1]- middle_coords[1]),(apex2_middle[2]- middle_coords[2]),(apex2_middle[3]- middle_coords[3])]

                if are_equal
                    if( ((xd - middle_coords[1])<1 && (yd - middle_coords[2])<1 && (zd - middle_coords[3])<1) || (full_data) )
                        push!(results, Dict("rel_coord" =>rel , 
                        # "triangle_points"=>[[rel...,submatrix_middle_tetr[ii, 4]] for ii in 1:3],
                        "triangle_points"=>triangle_points,
                        "triangle_points_middle"=>triangle_points_middle,
                        "apex1_middle"=>tetr_3d[middle_coords[1], middle_coords[2], middle_coords[3], j, 1, 1:4],#current sv center
                        "apex2_middle"=>tetr_3d[xd, yd, zd, j, 1, 1:4],#neighbour sv center
                        "apex1"=>apex1,#current sv center
                        "apex2"=>apex2,#neighbouring sv center
                        "num_tetr_neigh" => i, "current_tetr" => j))
                    end
                end
            end
        end
        return results
    end

    # Map over the coordinates and collect results
    res = map(coords) do (xd, yd, zd)
        check_and_collect(xd, yd, zd)
    end

    # Flatten the list of results
    return vcat(res...)
end




"""
based on output from check_triangles will create a plan how to access the information
about tetrahedrons that share base - we are looking just back axis so those that are before in x,y,z axis 
in order to avoid duplication first dimension will be index of a new point 
second will be indicating where to find in order 
    1)triangle_point1
    2)triangle_point2
    3)triangle_point3
    4)apex1
    5)apex2
third wil have 4 values where first 3 will indicate the x,y,z of control points and fourth the channel in control points 
where to find the coordinates of the control points used now ; Hovewer the coordinates would be relatinve to the current index 
so we will need to add the current index to the values from last dimension to get actual spot in the control points
"""
function get_plan_tensor_for_points_per_triangle(plan_tuples)
    #initialize the plan tensor
    plan_tensor=zeros(Int64,(length(plan_tuples),5,4))
    plan_tuples=sort!(plan_tuples, by = x -> x["current_tetr"])

    for i in 1:length(plan_tuples)
        plan=plan_tuples[i]
        plan_tensor[i,1,:]=plan["triangle_points"][1]
        plan_tensor[i,2,:]=plan["triangle_points"][2]
        plan_tensor[i,3,:]=plan["triangle_points"][3]
        plan_tensor[i,4,:]=[plan["apex1"]...,-1]# sv center of current supervoxel
        plan_tensor[i,5,:]=[plan["apex2"]...,-1]# sv center of neighbour supervoxel
    end
    return plan_tensor
end    


###### main

"""
    augment_flattened_triangles(flattened_triangles, plan_tensor, sv_centers)
Augments the `flattened_triangles` with information from `plan_tensor` and divides each tetrahedron into six new tetrahedrons.
# Arguments
- `flattened_triangles::Array{Int64, 4}`: 3D tensor where the first dimension is the index of the tetrahedron, the second is the index of the point (sv center is the first dimension, next 3 are indices of a base, and the last is additional space filled with -1), and the last dimension is x, y, z coordinates.
- `plan_tensor::Array{Int64, 3}`: 3D tensor where the first index is the plan_index, the next is the index of a point (first 3 indices indicate the points for the tetrahedron base, and the last has length 4 where the first 3 are relative x, y, z coordinates and the fourth is the channel that is not relative).
- `sv_centers::Vector{Int}`: List of supervoxel centers.
# Returns
- `flattened_triangles_augmented::Array{Int64, 4}`: Augmented list of tetrahedrons.
!!! requirees full plan_tensor - so with 48 entries

basically the sv center is base ind .+1
"""
function augment_flattened_triangles(tetrs, tetr_3d,plan_tuples_full,plan_tuples_sorted)
    
    # Initialize an empty dictionary to find dictionary based on tetr index
    dict_of_dicts = Dict{Int, Dict}()  
    # Iterate over each dictionary in the list
    for subdict in plan_tuples_full
        # Use the value under "current_tetr" as the key
        key = subdict["current_tetr"]
        # Assign the entire subdictionary to this key
        dict_of_dicts[key] = subdict
    end

    # Initialize an empty dictionary to find position (channel in control points out) based on tetr index 
    pos_index_dict = Dict{Any, Int}()

    # Iterate over each dictionary in the list with an index
    for (index, subdict) in enumerate(plan_tuples_sorted)
        # Add entries for both "current_tetr" and "num_tetr_neigh"
        pos_index_dict[subdict["current_tetr"]] = index
        pos_index_dict[subdict["num_tetr_neigh"]] = index
    end

    #get a list of primary indicies - if index is not here we need to reach out to the neighbour for appropriate point
    prim_indicies=map(el->el["current_tetr"] ,plan_tuples_sorted)

    tetrs_size=size(tetrs)
    new_tetr_size=(tetrs_size[1]*3,tetrs_size[2],tetrs_size[3])
    new_flat_tetrs=zeros(new_tetr_size)
    #iterating over first dimension of the tetrs
    
    
    
    Threads.@threads for ind_prim in 1:tetrs_size[1]
        # print("* $ind_prim *")
        #getting which tetrahedron in sv it is
        ind_tetr=((ind_prim-1)%48)+1
        channel_control_points=pos_index_dict[ind_tetr]
        curr_dict=dict_of_dicts[ind_tetr]
        t1,t2,t3=curr_dict["triangle_points"]

        new_ind_tetr_base=((ind_prim-1)*3)+1
        tetr_curr=tetrs[ind_prim,:,:]
        base_ind=tetr_curr[1,1:3].-1
        

        sv_center=tetr_curr[1,:]
        triang_1=[t1[1]+base_ind[1],t1[2]+base_ind[2],t1[3]+base_ind[3],t1[4]]
        triang_2=[t2[1]+base_ind[1],t2[2]+base_ind[2],t2[3]+base_ind[3],t2[4]]
        triang_3=[t3[1]+base_ind[1],t3[2]+base_ind[2],t3[3]+base_ind[3],t3[4]]
        

        
        dummy=tetr_curr[5,:]
        #weather it is base ind or not depends on weather we are looking on prev or next in axis
        base_ind_p=base_ind
        if(!(ind_tetr in prim_indicies))
            base_ind_p=base_ind+(curr_dict["apex2"].-1)
        end
        #new point we created using get_random_point_in_tetrs_kern
        new_point=[base_ind_p[1],base_ind_p[2],base_ind_p[3],channel_control_points+7]
        # new_point=[1,1,1,2]


        #populating with new data - we will always have the same sv center 
        #and the same 2 old triangle points and a new one 
        #we start from new_ind_tetr_base and we will add 1,2,3
        
        ###1
        to_add=0
        new_flat_tetrs[new_ind_tetr_base+to_add,1,:]=sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add,2,:]=triang_1
        new_flat_tetrs[new_ind_tetr_base+to_add,3,:]=triang_2
        new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=new_point
        new_flat_tetrs[new_ind_tetr_base+to_add,5,:]=dummy

        to_add=1
        new_flat_tetrs[new_ind_tetr_base+to_add,1,:]=sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add,2,:]=triang_1
        new_flat_tetrs[new_ind_tetr_base+to_add,3,:]=triang_3
        new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=new_point
        new_flat_tetrs[new_ind_tetr_base+to_add,5,:]=dummy

        to_add=2
        new_flat_tetrs[new_ind_tetr_base+to_add,1,:]=sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add,2,:]=triang_3
        new_flat_tetrs[new_ind_tetr_base+to_add,3,:]=triang_2
        new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=new_point
        new_flat_tetrs[new_ind_tetr_base+to_add,5,:]=dummy

    end

    return new_flat_tetrs
end


"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
and the radius of supervoxels will return the grid of points that will be used as centers of supervoxels 
and the intialize positions of the control points
is_points_per_triangle- indicates weather we additionally supply the plan tensor for the points per triangle unrolled
and weather we are going to use the points per triangle unrolled - need to include them in the flattened_triangles data 
"""
function initialize_centers_and_control_points(image_shape,radius,is_points_per_triangle=false)
    sv_centers,dims,diffs= get_sv_centers(radius,image_shape)
    flattened_triangles=get_flattened_triangle_data(dims)  
    if(is_points_per_triangle)
        
        
        #get plan tensor
        middle_coords=(Int(floor(dims[1]/2)),Int(floor(dims[2]/2)),Int(floor(dims[3]/2)))
        dims_tetr=dims.-2
        # dims_tetr=(12,12,12)
        tetr_3d=reshape(flattened_triangles,(48,dims_tetr[1],dims_tetr[2],dims_tetr[3],5,4))
        tetr_3d=permutedims(tetr_3d,(2,3,4,1,5,6))
        middle_tetr=tetr_3d[middle_coords[1],middle_coords[2],middle_coords[3],:,:,:]

        middle_coords=(3,2,4)
        middle_tetr=tetr_3d[middle_coords[1],middle_coords[2],middle_coords[3],:,2:4,1:4]
        plan_tuples_full=check_triangles(middle_coords, tetr_3d, middle_tetr,true)
        #sorted plan tuples is not full - so 24 entries
        plan_tuples=check_triangles(middle_coords, tetr_3d, middle_tetr,false)
        plan_tuples_sorted=sort(plan_tuples, by=x->x["current_tetr"])

        plan_tensor=get_plan_tensor_for_points_per_triangle(plan_tuples_sorted)        

        flattened_triangles_augmented=augment_flattened_triangles(flattened_triangles, tetr_3d,plan_tuples_full,plan_tuples_sorted)
        
        res= sv_centers,initialize_control_points(image_shape,radius,is_points_per_triangle),flattened_triangles_augmented,dims,plan_tensor
        return res
    else    
        res= sv_centers,initialize_control_points(image_shape,radius),flattened_triangles,dims
        return res
    end
end#initialize_centeris_and_control_points    
    