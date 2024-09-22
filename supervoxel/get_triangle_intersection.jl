using LinearAlgebra

# Define a point and a normal vector for the plane
plane_point = [1.0, 1.0, 1.0]
plane_normal = [0.0, 0.0, 1.0]

# Define the vertices of the triangle
triangle = [[0.0, 0.0, 0.0], [2.0, -1.0, 2.0], [1.0, 2.0, 3.0]]

# Function to find the intersection of a line segment with a plane
function line_plane_intersection(p1, p2, plane_point, plane_normal)
    u = p2 - p1
    w = p1 - plane_point
    d = dot(plane_normal, u)
    n = -dot(plane_normal, w)
    
    if abs(d) < 1e-6
        return nothing  # The line is parallel to the plane
    end
    
    sI = n / d
    if sI < 0 || sI > 1
        return nothing  # The intersection point is not within the segment
    end
    
    intersection = p1 + sI * u
    return intersection
end

# Find intersections of the triangle edges with the plane
intersections = []
for i in 1:3
    p1 = triangle[i]
    p2 = triangle[mod1(i+1, 3)]
    intersection = line_plane_intersection(p1, p2, plane_point, plane_normal)
    if intersection !== nothing
        push!(intersections, intersection)
    end
end

println("Intersection points: ", intersections)


function get_crossection(plane_axis::Int, d::Float32, triangle_arr::Matrix{Float32})
    vertices = Float32[]
    indices = UInt32[]
    index_counter = 0

    for i in 1:size(triangle_arr, 1)
        triangle = triangle_arr[i, :, :]
        intersection_points = []

        for j in 1:3
            for k in j+1:3
                v1 = triangle[j, :]
                v2 = triangle[k, :]

                t1 = v1[plane_axis]
                t2 = v2[plane_axis]

                if (t1 - d) * (t2 - d) < 0
                    t = (d - t1) / (t2 - t1)
                    intersection_point = v1 + t * (v2 - v1)
                    push!(intersection_points, intersection_point)
                end
            end
        end

        if length(intersection_points) == 2
            for point in intersection_points
                push!(vertices, point...)
                push!(indices, index_counter)
                index_counter += 1
            end
        end
    end

    return vertices, indices
end

# Example usage
triangle_arr = Float32[
    0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0;
    1.0 1.0 1.0; 2.0 1.0 1.0; 1.0 2.0 1.0
]
plane_axis = 1
d = Float32(0.24)

vertices, indices = get_crossection(plane_axis, d, triangle_arr)
println("Vertices: ", vertices)
println("Indices: ", indices)