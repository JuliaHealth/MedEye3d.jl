using ModernGL
using GLFW, Revise
includet("/media/jm/hddData/projects/MedEye3d.jl/supervoxel/main_super_voxel/get_polihydra.jl")

function initialize_window(width::Int, height::Int, title::String)
    if !GLFW.Init()
        error("Failed to initialize GLFW")
    end
    window = GLFW.CreateWindow(width, height, title)
    if window == C_NULL
        error("Failed to create GLFW window")
    end
    GLFW.MakeContextCurrent(window)
    glViewport(0, 0, width, height)
    return window
end


const vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in int polygonIndex;

flat out int polyIndex;

void main()
{
    gl_Position = vec4(position, 1.0);
    polyIndex = polygonIndex;
}
"""

"""
Fragment shader should take into account the poligon index that is indicating which parameters of sinusoids to use for texture generation.
main equation is :((sin(2 * π / (texture_bank_p[t,sin_i,4]*max_wavelength) * ((TexCoord[1]) * cos(texture_bank_p[t,sin_i,1]*2*π) + (TexCoord[2]) * cos(texture_bank_p[t,sin_i,2]*2*π) + (1.0) * cos(texture_bank_p[t,sin_i,3]*2*π)))+(sin_p[polyIndex,4] *max_amplitude))*(texture_bank_p[t,sin_i,5]*max_amplitude)*multiplier)*sin_p[polyIndex, t+5 ]
part of the parameters are stored in the texture_bank_p and part in sin_p arrays both should be encoded as two dimensional textures and passed to the shader 
polyIndex is intended to be integer value that is used to index the texture values from passed texture parameters  ; num_texture_banks and num_sinusoids_per_bank are used and should be loaded as uniforms
TexCoord - is coordinate of 2 dimensional texture that is covering current shape its values should be between beg_image, and end Image uniforms (for first dimension beg_image_x, end_image_x
, for second dimension beg_image_y, end_image_y) TexCoord[1] and TexCoord[2] apart from being in the specified range should take into account location in the whole window not only in the current shape 
"""

const fragment_shader_source = """
#version 330 core

uniform sampler1D tex_1d;
uniform sampler2D sin_p_tex;
uniform sampler3D texture_bank_tex;
uniform vec2 windowSize;
uniform int num_texture_banks;
uniform int num_sinusoids_per_bank;
uniform float max_wavelength;
uniform float max_amplitude;
uniform float multiplier;

flat in int polyIndex;
out vec4 outColor;

void main() {
    vec2 TexCoord = gl_FragCoord.xy / windowSize;
    float final_color = 0.0;
    
    for(int t = 0; t < num_texture_banks; t++) {
        for(int sin_i = 0; sin_i < num_sinusoids_per_bank; sin_i++) {
            // Access texture_bank_p parameters using 3D texture
            vec4 bank_params = texelFetch(texture_bank_tex, 
                                        ivec3(t, sin_i, 0), 0);
            float bank_amplitude = texelFetch(texture_bank_tex, 
                                           ivec3(t, sin_i, 4), 0).r;
            
            // Access sin_p parameters using 2D texture
            float sin_p_amplitude = texelFetch(sin_p_tex, 
                                             ivec2(polyIndex, 4), 0).r;
            float sin_p_multiplier = texelFetch(sin_p_tex, 
                                              ivec2(polyIndex, t+5), 0).r;
            
            float wave = sin(2.0 * 3.14159 / (bank_params.w * max_wavelength) * 
                           (TexCoord.x * cos(bank_params.x * 2.0 * 3.14159) + 
                            TexCoord.y * cos(bank_params.y * 2.0 * 3.14159) + 
                            1.0 * cos(bank_params.z * 2.0 * 3.14159)));
                            
            final_color += ((wave + (sin_p_amplitude * max_amplitude)) * 
                           (bank_amplitude * max_amplitude) * 
                           multiplier) * sin_p_multiplier;
        }
    }
    
    outColor = vec4(final_color, final_color, final_color, 1.0);
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


# Compile shaders and create a shader program
function create_shader_program()
    # Create and compile shaders
    vertex_shader = create_shader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = create_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

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

function prepare_data(all_res)
    vertices = Float32[]
    indices = UInt32[]
    colors = Float32[]
    polygon_indices = Int32[]
    vertex_offset = 0

    for (polygon_index, polygon) in all_res
        # Generate a unique color for each polygon
        color = Float32[
            sin(0.3 * polygon_index)*0.5+0.5,
            sin(0.5 * polygon_index)*0.5+0.5,
            sin(0.7 * polygon_index)*0.5+0.5
        ]
        append!(colors, color)

        for triangle in polygon
            for vertex in triangle
                append!(vertices, Float32.(vertex)...)
                push!(polygon_indices, polygon_index - 1)  # Zero-based indexing for shader
            end
            push!(indices, UInt32(vertex_offset), UInt32(vertex_offset + 1), UInt32(vertex_offset + 2))
            vertex_offset += 3
        end
    end

    return vertices, indices, colors, polygon_indices
end



function upload_data(vertices, indices, polygon_indices)
    VAO = Ref{GLuint}(0)
    VBO = Ref{GLuint}(0)
    EBO = Ref{GLuint}(0)
    PBO = Ref{GLuint}(0)

    glGenVertexArrays(1, VAO)
    glGenBuffers(1, VBO)
    glGenBuffers(1, EBO)
    glGenBuffers(1, PBO)

    glBindVertexArray(VAO[])

    # Vertex positions
    glBindBuffer(GL_ARRAY_BUFFER, VBO[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, C_NULL)
    glEnableVertexAttribArray(0)

    # Polygon indices for color
    glBindBuffer(GL_ARRAY_BUFFER, PBO[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(polygon_indices), polygon_indices, GL_STATIC_DRAW)
    glVertexAttribIPointer(1, 1, GL_INT, 0, C_NULL)
    glEnableVertexAttribArray(1)

    # Element indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)

    return VAO[]
end


function render(window, shader_program, VAO, colors, num_indices, tex_1d, sin_p_tex, texture_bank_tex)
    glUseProgram(shader_program)

    # Bind textures to different texture units
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_1D, tex_1d)
    glUniform1i(glGetUniformLocation(shader_program, "tex_1d"), 0)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, sin_p_tex)
    glUniform1i(glGetUniformLocation(shader_program, "sin_p_tex"), 1)

    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_3D, texture_bank_tex)
    glUniform1i(glGetUniformLocation(shader_program, "texture_bank_tex"), 2)

    # Set colors uniform
    color_location = glGetUniformLocation(shader_program, "colors")
    glUniform3fv(color_location, length(colors) ÷ 3, colors)

    # Set window size uniform
    window_size_location = glGetUniformLocation(shader_program, "windowSize")
    width, height = GLFW.GetWindowSize(window)
    glUniform2f(window_size_location, Float32(width), Float32(height))

    while !GLFW.WindowShouldClose(window)
        GLFW.PollEvents()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, C_NULL)

        GLFW.SwapBuffers(window)
    end

    glDeleteVertexArrays(1, Ref(VAO))
    glDeleteProgram(shader_program)
    GLFW.Terminate()
end


function main(all_res, sv_means,windowWidth,windowHeight,texture_bank_p,sin_p)
    window = initialize_window(windowWidth, windowHeight, "Polygon Rendering")
    shader_program = create_shader_program()
    vertices, indices, colors, polygon_indices = prepare_data(all_res)
#     # ... more polygons
# ]

h5_path_b = "/media/jm/hddData/projects/MedEye3d.jl/docs/src/data/locc.h5"
fb = h5open(h5_path_b, "r")
#we want only external triangles so we ignore the sv center
# we also ignore interpolated variance value here
tetr_dat = fb["tetr_dat"][:, 2:4, 1:3, 1]
out_sampled_points = fb["out_sampled_points"]

sv_means = get_sv_mean(out_sampled_points)[:, 1]
sv_means = sv_means .- minimum(sv_means)
sv_means = sv_means ./ maximum(sv_means)
sv_means = Float32.(sv_means)

axis = 3
plane_dist = 25.0
radiuss = (Float32(4.5), Float32(4.5), Float32(4.5))

all_res = main_get_poligon_data(tetr_dat, axis, plane_dist, radiuss)



function get_num_tetr_in_sv()
    return 48
end



### prepare data for texture
sizz_out = size(out_sampled_points)
batch_size = sizz_out[end]
n_tetr = get_num_tetr_in_sv()
num_sv = Int(round(sizz_out[1] / n_tetr))

texture_bank_p = rand(Float32, num_texture_banks, num_sinusoids_per_bank, 5)

sin_p = rand(Float32, sizz_out[1], num_texture_banks + 6) .* 2
sin_p_a = sin_p[:, 1:5]
sin_p_b = softmax(sin_p[:, 6:end], dims=2)
sin_p = cat(sin_p_a, sin_p_b, dims=2)



windowWidth,windowHeight=800,800
main(all_res, sv_means,windowWidth,windowHeight,texture_bank_p,sin_p)