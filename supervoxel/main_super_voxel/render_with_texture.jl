using ModernGL
using GLFW, Revise, Lux
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
out vec2 TexCoord;

void main() {
    gl_Position = vec4(position, 1.0);
    polyIndex = polygonIndex;
    TexCoord = (position.xy + 1.0) / 2.0;  // Map position to [0, 1] range
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

uniform sampler2D texture_bank_p;      // Parameters: alpha, beta, gamma, wavelength
uniform sampler2D texture_bank_p_amp;  // Parameter: amplitude
uniform sampler2D sin_p;               // Parameters per polygon index
flat in int polyIndex;

uniform int num_texture_banks;
uniform int num_sinusoids_per_bank;
uniform float max_wavelength;
uniform float max_amplitude;
uniform float multiplier;
uniform float beg_image_x;
uniform float end_image_x;
uniform float beg_image_y;
uniform float end_image_y;

in vec2 TexCoord;
out vec4 outColor;

void main() {
    float value = 0.0;
    float pi = 3.14159265;

    // Map TexCoord to global image coordinates
    float x = mix(beg_image_x, end_image_x, TexCoord.x);
    float y = mix(beg_image_y, end_image_y, TexCoord.y);
    float z = 1.0;

    vec2 sin_p_size = textureSize(sin_p, 0);
    float sin_p_base = texture(sin_p, vec2((4.5) / sin_p_size.x, (polyIndex + 0.5) / sin_p_size.y)).r * max_amplitude;

    for (int t = 0; t < num_texture_banks; t++) {
        for (int sin_i = 0; sin_i < num_sinusoids_per_bank; sin_i++) {
            vec2 tex_coord = vec2((sin_i + 0.5) / num_sinusoids_per_bank, (t + 0.5) / num_texture_banks);

            // Fetch parameters
            vec4 params = texture(texture_bank_p, tex_coord);
            float amplitude_param = texture(texture_bank_p_amp, tex_coord).r;
            float alpha = params.r * 2.0 * pi;
            float beta = params.g * 2.0 * pi;
            float gamma = params.b * 2.0 * pi;
            float wavelength = params.a * max_wavelength;
            float amplitude = amplitude_param * max_amplitude;

            // Fetch sin_p coefficient
            float sin_p_index = float(t * num_sinusoids_per_bank + sin_i + 5) + 0.5;
            float sin_p_coeff = texture(sin_p, vec2(sin_p_index / sin_p_size.x, (polyIndex + 0.5) / sin_p_size.y)).r;

            // Compute phase
            float phase = x * cos(alpha) + y * cos(beta) + z * cos(gamma);

            // Main equation
            value += ((sin(2.0 * pi / wavelength * phase) + sin_p_base) * amplitude * multiplier) * sin_p_coeff;
        }
    }

    value = value * 0.5 + 0.5;
    outColor = vec4(value, value, value, 1.0);
}
"""

function create_shader(source, shader_type)
    shader = glCreateShader(shader_type)  # Add this line to create the shader
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

function prepare_data(all_res, sin_p, texture_bank_p)
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

    return vertices, indices, colors, polygon_indices, sin_p, texture_bank_p
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


function render(window, shader_program, VAO, num_indices, sin_p_texture, texture_bank_p_texture, texture_bank_p_texture_amp)
    glUseProgram(shader_program)

    # Set uniform values
    glUniform1i(glGetUniformLocation(shader_program, "num_texture_banks"), num_texture_banks)
    glUniform1i(glGetUniformLocation(shader_program, "num_sinusoids_per_bank"), num_sinusoids_per_bank)
    glUniform1f(glGetUniformLocation(shader_program, "max_wavelength"), 10.0f0)  # Adjust as needed
    glUniform1f(glGetUniformLocation(shader_program, "max_amplitude"), 1.0f0)
    glUniform1f(glGetUniformLocation(shader_program, "multiplier"), 1.0f0)
    glUniform1f(glGetUniformLocation(shader_program, "beg_image_x"), -1.0f0)
    glUniform1f(glGetUniformLocation(shader_program, "end_image_x"), 1.0f0)
    glUniform1f(glGetUniformLocation(shader_program, "beg_image_y"), -1.0f0)
    glUniform1f(glGetUniformLocation(shader_program, "end_image_y"), 1.0f0)

    # Bind textures
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, sin_p_texture)
    glUniform1i(glGetUniformLocation(shader_program, "sin_p"), 0)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, texture_bank_p_texture)
    glUniform1i(glGetUniformLocation(shader_program, "texture_bank_p"), 1)

    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, texture_bank_p_texture_amp)
    glUniform1i(glGetUniformLocation(shader_program, "texture_bank_p_amp"), 2)

    while !GLFW.WindowShouldClose(window)
        GLFW.PollEvents()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, C_NULL)

        GLFW.SwapBuffers(window)
        print("*****")
    end

    glDeleteVertexArrays(1, Ref(VAO))
    glDeleteProgram(shader_program)
    GLFW.Terminate()
end


function main(all_res, sv_means, sin_p, texture_bank_p)
    window = initialize_window(800, 600, "Polygon Rendering")
    shader_program = create_shader_program()
    vertices, indices, colors, polygon_indices, sin_p_data, texture_bank_p_data = prepare_data(all_res, sin_p, texture_bank_p)
    VAO = upload_data(vertices, indices, polygon_indices)
    num_indices = length(indices)

    # Calculate texture dimensions
    width_sin_p = size(sin_p, 2)  # number of columns
    height_sin_p = size(sin_p, 1)  # number of rows

    width_texture_bank_p = size(texture_bank_p, 2)  # num_sinusoids_per_bank
    height_texture_bank_p = size(texture_bank_p, 1)  # num_texture_banks

    # Convert texture data to the correct format for OpenGL
    sin_p_tex_data = zeros(Float32, height_sin_p, width_sin_p, 1)
    for i in 1:height_sin_p
        for j in 1:width_sin_p
            sin_p_tex_data[i, j, 1] = sin_p[i, j]  # R component
        end
    end

    texture_bank_p_tex_data = zeros(Float32, height_texture_bank_p, width_texture_bank_p, 5)
    for i in 1:height_texture_bank_p
        for j in 1:width_texture_bank_p
            for k in 1:5  # First 5 parameters of texture_bank_p
                texture_bank_p_tex_data[i, j, k] = texture_bank_p[i, j, k]
            end
        end
    end

    # Split texture_bank_p into two textures due to texture format limitations
    # First texture with parameters 1-4
    texture_bank_p_tex_data1 = texture_bank_p_tex_data[:, :, 1:4]
    # Second texture with parameter 5 (amplitude)
    texture_bank_p_tex_data2 = texture_bank_p_tex_data[:, :, 5]

    # Initialize a 1D texture with sv_means data
    tex_1d = Ref{GLuint}()
    glGenTextures(1, tex_1d)
    glBindTexture(GL_TEXTURE_1D, tex_1d[])
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, length(sv_means), 0, GL_RED, GL_FLOAT, pointer(sv_means))

    # Prepare sin_p and texture_bank_p textures
    sin_p_texture = Ref{GLuint}()
    glGenTextures(1, sin_p_texture)
    glBindTexture(GL_TEXTURE_2D, sin_p_texture[])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width_sin_p, height_sin_p, 0, GL_RED, GL_FLOAT, pointer(sin_p_tex_data))

    texture_bank_p_texture = Ref{GLuint}()
    glGenTextures(1, texture_bank_p_texture)
    glBindTexture(GL_TEXTURE_2D, texture_bank_p_texture[])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_texture_bank_p, height_texture_bank_p, 0, GL_RGBA, GL_FLOAT, pointer(texture_bank_p_tex_data1))

    texture_bank_p_texture_amp = Ref{GLuint}()
    glGenTextures(1, texture_bank_p_texture_amp)
    glBindTexture(GL_TEXTURE_2D, texture_bank_p_texture_amp[])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width_texture_bank_p, height_texture_bank_p, 0, GL_RED, GL_FLOAT, pointer(texture_bank_p_tex_data2))

    # Enable GL_BLEND for proper texture blending
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    render(window, shader_program, VAO, num_indices, sin_p_texture[], texture_bank_p_texture[], texture_bank_p_texture_amp[])
end

# all_res = [
#     [  # Polygon 1
#         [ (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0) ],  # Triangle 1
#         # ... more triangles
#     ],
#     [  # Polygon 2
#         [ (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0) ],  # Triangle 1
#         # ... more triangles
#     ],
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
num_texture_banks = 32
num_sinusoids_per_bank = 4
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
#TODO we use sin_p without batch_size




main(all_res, sv_means, sin_p, texture_bank_p)


