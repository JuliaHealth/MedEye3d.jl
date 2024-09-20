"""

@workspace you are OPENGL graphics and Julia programming expert. Given array of tetrahedrons that are encored as  3 dimensional array where first dimension is index of tetrahedron secod is index of one of the 4 points in tetrahedron and last dimension is encoding x,y,z coordinates. Get a crossection of all tetrahedrons that are crossing a plane that is perpendicular to the x or y or z axis and is in distance d from the center. Render then all lines and points that indicate the crossection of the borders of tetrahedrons on this plane using Opengl. Work step by step implementing each step as a separate function.  
"""





using LinearAlgebra
using ModernGL
import ModernGL
using GLFW

# Define OpenGL functions
ModernGL.@glfunc glVertex3f(x::GLfloat, y::GLfloat, z::GLfloat)::Cvoid
# ModernGL.@glfunc glClear(mask::GLbitfield)::Cvoid
ModernGL.@glfunc glClearColor(red::GLfloat, green::GLfloat, blue::GLfloat, alpha::GLfloat)::Cvoid
ModernGL.@glfunc glEnd()::Cvoid
ModernGL.@glfunc glDrawArrays(mode::GLenum, first::GLint, count::GLsizei)::Cvoid

struct Tetrahedron
    points::Array{Float64, 2}  # 4x3 array for 4 points with x, y, z coordinates
end

function compute_intersection(tetrahedrons::Array{Tetrahedron, 1}, axis::Symbol, d::Float64)
    axis_index = Dict(:x => 1, :y => 2, :z => 3)[axis]
    intersections = []
    for tetra in tetrahedrons
        for i in 1:4
            for j in i+1:4
                p1 = tetra.points[i, :]
                p2 = tetra.points[j, :]
                if (p1[axis_index] - d) * (p2[axis_index] - d) < 0
                    t = (d - p1[axis_index]) / (p2[axis_index] - p1[axis_index])
                    intersection_point = p1 + t * (p2 - p1)
                    push!(intersections, intersection_point)
                end
            end
        end
    end
    return intersections
end

function render_intersection(intersections)
    if !GLFW.Init()
        error("Failed to initialize GLFW")
    end

    window = GLFW.CreateWindow(800, 600, "Tetrahedron Cross-Section")
    if window == C_NULL
        error("Failed to create GLFW window")
    end

    GLFW.MakeContextCurrent(window)
    # ModernGL.LoadFunctions()  # Load OpenGL functions

    while !GLFW.WindowShouldClose(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for i in 1:2:length(intersections)
            glVertex3f(intersections[i][1], intersections[i][2], intersections[i][3])
            glVertex3f(intersections[i+1][1], intersections[i+1][2], intersections[i+1][3])
        end
        glEnd()
        GLFW.SwapBuffers(window)
        GLFW.PollEvents()
    end

    GLFW.Terminate()
end

# Example usage
tetrahedrons = [
    Tetrahedron([0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
    Tetrahedron([1.0 1.0 1.0; 2.0 1.0 1.0; 1.0 2.0 1.0; 1.0 1.0 2.0])
]

axis = :x
d = 0.5
intersections = compute_intersection(tetrahedrons, axis, d)
render_intersection(intersections)




# @workspace you are OPENGL graphics and Julia programming expert. Given array of tetrahedrons that are encored as  3 dimensional array where first dimension is index of tetrahedron secod is index of one of the 4 points in tetrahedron and last dimension is encoding x,y,z coordinates. Get a crossection of all tetrahedrons that are crossing a plane that is perpendicular to the x or y or z axis and is in distance d from the center. Render then all lines and points that indicate the crossection of the borders of tetrahedrons on this plane using Opengl. Work step by step implementing each step as a separate function.  
# """
# using LinearAlgebra
# using ModernGL
# using GLFW

# struct Tetrahedron
#     points::Array{Float64, 2}  # 4x3 array for 4 points with x, y, z coordinates
# end





# function compute_intersection(tetrahedrons::Array{Tetrahedron, 1}, axis::Symbol, d::Float64)
#     axis_index = Dict(:x => 1, :y => 2, :z => 3)[axis]
#     intersections = []
#     for tetra in tetrahedrons
#         for i in 1:4
#             for j in i+1:4
#                 p1 = tetra.points[i, :]
#                 p2 = tetra.points[j, :]
#                 if (p1[axis_index] - d) * (p2[axis_index] - d) < 0
#                     t = (d - p1[axis_index]) / (p2[axis_index] - p1[axis_index])
#                     intersection_point = p1 + t * (p2 - p1)
#                     push!(intersections, intersection_point)
#                 end
#             end
#         end
#     end
#     return intersections
# end



# function render_intersection(intersections::Array{Array{Float64, 1}})
#     if !GLFW.Init()
#         error("Failed to initialize GLFW")
#     end

#     window = GLFW.CreateWindow(800, 600, "Tetrahedron Cross-Section")
#     if window == C_NULL
#         error("Failed to create GLFW window")
#     end

#     GLFW.MakeContextCurrent(window)
#     GL.loadfunctions()

#     while !GLFW.WindowShouldClose(window)
#         GL.Clear(GL.COLOR_BUFFER_BIT | GL.DEPTH_BUFFER_BIT)
#         GL.Begin(GL.LINES)
#         for i in 1:2:length(intersections)
#             GL.Vertex3f(intersections[i][1], intersections[i][2], intersections[i][3])
#             GL.Vertex3f(intersections[i+1][1], intersections[i+1][2], intersections[i+1][3])
#         end
#         GL.End()
#         GLFW.SwapBuffers(window)
#         GLFW.PollEvents()
#     end

#     GLFW.Terminate()
# end




# # Example usage
# tetrahedrons = [
#     Tetrahedron([0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
#     Tetrahedron([1.0 1.0 1.0; 2.0 1.0 1.0; 1.0 2.0 1.0; 1.0 1.0 2.0])
# ]

# axis = :x
# d = 0.5
# intersections = compute_intersection(tetrahedrons, axis, d)
# render_intersection(intersections)
# """ 
# line """if (p1[axis] - d) * (p2[axis] - d) < 0""" of function "compute_intersection" give error """invalid index: :x of type Symbol""" analyse all step by step and correct the function


