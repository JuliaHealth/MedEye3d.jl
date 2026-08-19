with open("src/preprocessing/BoneSubsegmentation.jl", "r") as f:
    text = f.read()

import re

old = r"""    # Extract true cortical surface using a 3D flood fill from the bounding box edges\.
    # We treat BOTH bone and lesion as obstacles to prevent flood filling the marrow if the lesion broke the cortical shell!
    cx_dim, cy_dim, cz_dim = crop_dims
    outside = zeros\(Bool, crop_dims\)
    q = Tuple\{Int, Int, Int\}\[\]
    for i in 1:cx_dim, j in 1:cy_dim
        push!\(q, \(i, j, 1\)\)
        push!\(q, \(i, j, cz_dim\)\)
    end
    for i in 1:cx_dim, k in 1:cz_dim
        push!\(q, \(i, 1, k\)\)
        push!\(q, \(i, cy_dim, k\)\)
    end
    for j in 1:cy_dim, k in 1:cz_dim
        push!\(q, \(1, j, k\)\)
        push!\(q, \(cx_dim, j, k\)\)
    end
    
    while !isempty\(q\)
        i, j, k = pop!\(q\)
        if !outside\[i, j, k\] && !crop_bone_bool\[i, j, k\] && !crop_lesion\[i, j, k\]
            outside\[i, j, k\] = true
            if i > 1 && !outside\[i-1, j, k\] push!\(q, \(i-1, j, k\)\) end
            if i < cx_dim && !outside\[i\+1, j, k\] push!\(q, \(i\+1, j, k\)\) end
            if j > 1 && !outside\[i, j-1, k\] push!\(q, \(i, j-1, k\)\) end
            if j < cy_dim && !outside\[i, j\+1, k\] push!\(q, \(i, j\+1, k\)\) end
            if k > 1 && !outside\[i, j, k-1\] push!\(q, \(i, j, k-1\)\) end
            if k < cz_dim && !outside\[i, j, k\+1\] push!\(q, \(i, j, k\+1\)\) end
        end
    end
    
    # Bone \+ Marrow = everything not outside
    solid_bone = \.!outside
    
    crop_cortical = zeros\(Bool, crop_dims\)
    @inbounds for k in 1:cz_dim, j in 1:cy_dim, i in 1:cx_dim
        if solid_bone\[i, j, k\]
            # It's cortical surface if it touches the outside
            if i == 1 \|\| i == cx_dim \|\| j == 1 \|\| j == cy_dim \|\| k == 1 \|\| k == cz_dim \|\|
               outside\[i-1, j, k\] \|\| outside\[i\+1, j, k\] \|\|
               outside\[i, j-1, k\] \|\| outside\[i, j\+1, k\] \|\|
               outside\[i, j, k-1\] \|\| outside\[i, j, k\+1\]
                crop_cortical\[i, j, k\] = true
            end
        end
    end
    
    # Bone marrow is the interior core of the solid bone
    crop_marrow = solid_bone \.& \.!crop_cortical"""

new = """    cx_dim, cy_dim, cz_dim = crop_dims
    
    # Apply a fast 3D morphological closing (Dilation -> Erosion) to fill internal trabecular holes 
    # without breaking if the marrow is exposed at the bounding box edges.
    dilated = copy(crop_bone_bool)
    @inbounds for k in 1:cz_dim, j in 1:cy_dim, i in 1:cx_dim
        if crop_bone_bool[i, j, k] || crop_lesion[i, j, k]
            for dk in -1:1, dj in -1:1, di in -1:1
                ni, nj, nk = i+di, j+dj, k+dk
                if 1 <= ni <= cx_dim && 1 <= nj <= cy_dim && 1 <= nk <= cz_dim
                    dilated[ni, nj, nk] = true
                end
            end
        end
    end
    
    closed_bone = copy(dilated)
    @inbounds for k in 1:cz_dim, j in 1:cy_dim, i in 1:cx_dim
        if dilated[i, j, k]
            is_edge = false
            for dk in -1:1, dj in -1:1, di in -1:1
                ni, nj, nk = i+di, j+dj, k+dk
                if 1 <= ni <= cx_dim && 1 <= nj <= cy_dim && 1 <= nk <= cz_dim
                    if !dilated[ni, nj, nk]
                        is_edge = true
                        break
                    end
                end
            end
            if is_edge
                closed_bone[i, j, k] = false
            end
        end
    end

    # Extract true cortical surface using 6-connected neighbor check on the closed solid bone
    crop_cortical = zeros(Bool, crop_dims)
    @inbounds for k in 1:cz_dim, j in 1:cy_dim, i in 1:cx_dim
        if closed_bone[i, j, k]
            if i == 1 || i == cx_dim || j == 1 || j == cy_dim || k == 1 || k == cz_dim ||
               !closed_bone[i-1, j, k] || !closed_bone[i+1, j, k] ||
               !closed_bone[i, j-1, k] || !closed_bone[i, j+1, k] ||
               !closed_bone[i, j, k-1] || !closed_bone[i, j, k+1]
                crop_cortical[i, j, k] = true
            end
        end
    end
    
    # Bone marrow is the interior core of the closed bone
    crop_marrow = closed_bone .& .!crop_cortical"""

match = re.search(old, text)
if match:
    text = text[:match.start()] + new + text[match.end():]
    with open("src/preprocessing/BoneSubsegmentation.jl", "w") as f:
        f.write(text)
    print("Patched BoneSubsegmentation.jl!")
else:
    print("Not found!")
