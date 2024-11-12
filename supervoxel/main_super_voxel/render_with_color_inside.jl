using ModernGL
using GLFW
using LinearAlgebra, Revise
includet("/media/jm/hddData/projects/MedEye3d.jl/supervoxel/main_super_voxel/get_polihydra.jl")

# Vertex shader source
const vertex_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec3 fragColor;

void main() {
    gl_Position = vec4(position, 1.0);
    fragColor = color;
}
"""

# Fragment shader source
const fragment_source = """
#version 330 core
in vec3 fragColor;
out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
"""

function create_shader(source, shader_type)
    shader = glCreateShader(shader_type)
    glShaderSource(shader, 1, convert(Ptr{UInt8}, pointer([convert(Ptr{GLchar}, pointer(source))])), C_NULL)
    glCompileShader(shader)
    
    # Check for compilation errors
    status = GLint[0]
    glGetShaderiv(shader, GL_COMPILE_STATUS, status)
    if status[] == GL_FALSE
        maxlength = 1024
        buffer = zeros(GLchar, maxlength)
        sizei = GLsizei[0]
        glGetShaderInfoLog(shader, maxlength, sizei, buffer)
        error("Shader compilation failed: ", unsafe_string(pointer(buffer), sizei[]))
    end
    
    return shader
end

function create_program()
    # Create and compile shaders
    vertex_shader = create_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = create_shader(fragment_source, GL_FRAGMENT_SHADER)
    
    # Create and link program
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    # Check for linking errors
    status = GLint[0]
    glGetProgramiv(program, GL_LINK_STATUS, status)
    if status[] == GL_FALSE
        maxlength = 1024
        buffer = zeros(GLchar, maxlength)
        sizei = GLsizei[0]
        glGetProgramInfoLog(program, maxlength, sizei, buffer)
        error("Program linking failed: ", unsafe_string(pointer(buffer), sizei[]))
    end

    # Clean up shaders
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program
end

function render_polyhedrons(polyhedron_dict)
    # Initialize GLFW
    if !GLFW.Init()
        error("Failed to initialize GLFW")
    end

    # Create window
    window = GLFW.CreateWindow(800, 600, "Polyhedrons Visualization")
    if window == C_NULL
        GLFW.Terminate()
        error("Failed to create GLFW window")
    end

    GLFW.MakeContextCurrent(window)

    # Create shader program
    program = create_program()
    glUseProgram(program)

    # Main render loop
    while !GLFW.WindowShouldClose(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # For each supervoxel index and its points
        for (sv_index, points) in polyhedron_dict
            print(" ******* **** ")
            # Generate color based on supervoxel index
            color = Float32[
                sin(sv_index * 0.3) * 0.5 + 0.5,
                sin(sv_index * 0.5) * 0.5 + 0.5,
                sin(sv_index * 0.7) * 0.5 + 0.5
            ]

            # Convert points to Float32 array
            vertices = Float32[]
            colors = Float32[]
            
            # Add vertices and colors
            for i in 1:size(points, 1)
                append!(vertices, points[i, :])
                append!(colors, color)
            end

            # Create and bind VAO
            vao = GLuint[0]
            glGenVertexArrays(1, vao)
            glBindVertexArray(vao[1])

            # Create and bind vertex buffer
            vbo = GLuint[0]
            glGenBuffers(1, vbo)
            glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, C_NULL)
            glEnableVertexAttribArray(0)

            # Create and bind color buffer
            cbo = GLuint[0]
            glGenBuffers(1, cbo)
            glBindBuffer(GL_ARRAY_BUFFER, cbo[1])
            glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, C_NULL)
            glEnableVertexAttribArray(1)

            # Draw the polyhedron
            glDrawArrays(GL_TRIANGLE_STRIP, 0, length(vertices) ÷ 3)

            # Cleanup
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            glDeleteBuffers(1, vbo)
            glDeleteBuffers(1, cbo)
            glDeleteVertexArrays(1, vao)
        end

        GLFW.SwapBuffers(window)
        GLFW.PollEvents()
    end

    GLFW.DestroyWindow(window)
    GLFW.Terminate()
end


h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"

axis=3
plane_dist=19.0
radiuss = (Float32(4.5), Float32(4.5), Float32(4.5))

fb = h5open(h5_path_b, "r")
#we want only external triangles so we ignore the sv center
# we also ignore interpolated variance value here
tetr_dat=fb["tetr_dat"][:,2:4,1:3,:]
#we need to reshape the data to have sv index as first dimension and index of tetrahedron in sv as a second

tetr_s=size(tetr_dat)
batch_size=tetr_s[end]
tetr_dat = reshape(tetr_dat, (get_num_tetr_in_sv(), Int(round(tetr_s[1] / get_num_tetr_in_sv())), tetr_s[2], tetr_s[3], batch_size))
intersection_points_aug=get_intersection_point_augmented(tetr_dat, axis, plane_dist)

minimum(intersection_points_aug[:,1])
maximum(intersection_points_aug[:,1])
minimum(intersection_points_aug[:,2])
maximum(intersection_points_aug[:,2])
minimum(intersection_points_aug[:,3])
maximum(intersection_points_aug[:,3])


polyhedron_dict=create_supervoxel_dict(intersection_points_aug)


render_polyhedrons(polyhedron_dict)