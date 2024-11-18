using ModernGL
using GLFW, Revise,Lux
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


TODO do sth like here - so get coords between 0 and 1 
// Convert polyIndex to texture coordinate between 0.0 and 1.0
float coord = float(polyIndex) / (textureSize(tex_1d, 0) - 1);
float amplitude = texture(tex_1d, coord).r;


const fragment_shader_source = """
#version 330 core

uniform sampler1D tex_1d;
uniform sampler2D sin_p_tex;
uniform sampler3D texture_bank_tex;
uniform vec2 windowSize;
uniform float max_wavelength;
uniform float max_amplitude;
uniform float multiplier;
uniform int num_texture_banks;
uniform int num_sinusoids_per_bank;

flat in int polyIndex;
out vec4 outColor;

void main() {
    vec2 TexCoord = gl_FragCoord.xy / windowSize;


     // Get screen coordinates
    vec2 screenPos = gl_FragCoord.xy / windowSize;  // Now using uniform windowSize
    
    // Convert polyIndex to texture coordinate between 0.0 and 1.0
    float coord = float(polyIndex) / (textureSize(tex_1d, 0) - 1);
    
    // Create 2D sinusoid pattern
    int t=5;
    int sin_i=1;

    //float amplitude = texelFetch(sin_p_tex, ivec2(polyIndex, 4), 0).r;
    float amplitude =1.0;
    float wavelength = texelFetch(texture_bank_tex, ivec3(t, sin_i, 4), 0).r;
    

    float frequency = 1 / wavelength;  // Adjust this value to change wave frequency
    float wave = amplitude * sin(frequency * screenPos.x) * sin(frequency * screenPos.y);
    
    // Scale wave to [0,1] range
    wave = wave * 0.5 + 0.5;
    

    //((sin(2 * π / (texture_bank_p[t,sin_i,4]*max_wavelength) * ((TexCoord[1]) * cos(texture_bank_p[t,sin_i,1]*2*π) + (TexCoord[2]) * cos(texture_bank_p[t,sin_i,2]*2*π) + (1.0) * cos(texture_bank_p[t,sin_i,3]*2*π)))+(sin_p[polyIndex,4] *max_amplitude))*(texture_bank_p[t,sin_i,5]*max_amplitude)*multiplier)*sin_p[polyIndex, t+5 ]


    outColor = vec4(wavelength, wavelength, wavelength, 1.0);
    //outColor = vec4(wave, wave, wave, 1.0);

    //outColor = vec4(final_color, final_color, final_color, 1.0);
}
"""


# float final_color = 0.0;
    
# for(int t = 0; t < num_texture_banks; t++) {
#     for(int sin_i = 0; sin_i < num_sinusoids_per_bank; sin_i++) {  // Fixed variable
#         vec4 bank_params = texelFetch(texture_bank_tex, ivec3(t, sin_i, 0), 0);
#         float wavelength = texelFetch(texture_bank_tex, ivec3(t, sin_i, 4), 0).r;
        
#         float sin_p_amplitude = texelFetch(sin_p_tex, ivec2(polyIndex, 4), 0).r;
#         float sin_p_multiplier = texelFetch(sin_p_tex, ivec2(polyIndex, t+5), 0).r;
        
#         float wave = sin(2.0 * 3.14159 / (wavelength * max_wavelength) * 
#                        (TexCoord.x * cos(bank_params.x * 2.0 * 3.14159) + 
#                         TexCoord.y * cos(bank_params.y * 2.0 * 3.14159) + 
#                         cos(bank_params.z * 2.0 * 3.14159)));
        
#         final_color += ((wave + (sin_p_amplitude * max_amplitude)) * 
#                       (bank_params.w * max_amplitude) * 
#                       multiplier) * sin_p_multiplier;
#     }
# }

# final_color = final_color * 0.5 + 0.5;  // normalize to [0,1]


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

    # Set window size uniform
    window_size_location = glGetUniformLocation(shader_program, "windowSize")
    width, height = GLFW.GetWindowSize(window)
    glUniform2f(window_size_location, Float32(width), Float32(height))

    # Bind all textures
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_1D, tex_1d)
    glUniform1i(glGetUniformLocation(shader_program, "tex_1d"), 0)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, sin_p_tex)
    glUniform1i(glGetUniformLocation(shader_program, "sin_p_tex"), 1)

    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_3D, texture_bank_tex)
    glUniform1i(glGetUniformLocation(shader_program, "texture_bank_tex"), 2)

    while !GLFW.WindowShouldClose(window)
        print("****** ")
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


function main(all_res, sv_means, windowWidth, windowHeight, texture_bank_p, sin_p)
    window = initialize_window(windowWidth, windowHeight, "Polygon Rendering")
    shader_program = create_shader_program()
    vertices, indices, colors, polygon_indices = prepare_data(all_res)
    VAO = upload_data(vertices, indices, polygon_indices)
    num_indices=length(indices)
    # Create textures
    tex_1d = Ref{GLuint}()
    sin_p_tex = Ref{GLuint}()
    texture_bank_tex = Ref{GLuint}()
    
    # Setup 1D texture
    glGenTextures(1, tex_1d)
    glBindTexture(GL_TEXTURE_1D, tex_1d[])
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, length(sv_means), 0, GL_RED, GL_FLOAT, pointer(sv_means))

    # Setup 2D texture for sin_p
    glGenTextures(1, sin_p_tex)
    glBindTexture(GL_TEXTURE_2D, sin_p_tex[])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, size(sin_p, 2), size(sin_p, 1), 0, GL_RED, GL_FLOAT, pointer(sin_p))

    # Setup 3D texture for texture_bank_p
    glGenTextures(1, texture_bank_tex)
    glBindTexture(GL_TEXTURE_3D, texture_bank_tex[])
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, size(texture_bank_p, 3), size(texture_bank_p, 2), 
                 size(texture_bank_p, 1), 0, GL_RED, GL_FLOAT, pointer(texture_bank_p))

    # Set additional uniforms
    glUseProgram(shader_program)
    glUniform1i(glGetUniformLocation(shader_program, "num_texture_banks"), size(texture_bank_p, 1))
    glUniform1i(glGetUniformLocation(shader_program, "num_sinusoids_per_bank"), size(texture_bank_p, 2))
    glUniform1f(glGetUniformLocation(shader_program, "max_wavelength"), 100.0)
    glUniform1f(glGetUniformLocation(shader_program, "max_amplitude"), 1.0)
    glUniform1f(glGetUniformLocation(shader_program, "multiplier"), 1.0)

    render(window, shader_program, VAO, colors, num_indices, tex_1d[], sin_p_tex[], texture_bank_tex[])
end

# Example usage (replace `all_res` with your actual data)
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
num_texture_banks=32 
num_sinusoids_per_bank=4
texture_bank_p = rand(Float32, num_texture_banks, num_sinusoids_per_bank, 5)

sin_p = rand(Float32, sizz_out[1], num_texture_banks + 6) .* 2
sin_p_a = sin_p[:, 1:5]
sin_p_b = softmax(sin_p[:, 6:end], dims=2)
sin_p = cat(sin_p_a, sin_p_b, dims=2)



windowWidth,windowHeight=800,800
main(all_res, sv_means,windowWidth,windowHeight,texture_bank_p,sin_p)