using Revise
using ModernGL
using GLFW,HDF5
using GeometryTypes
using Images
using TensorBoardLogger,Hyperopt
includet("/home/jakubmitura/projects/MedEye3d.jl/supervoxel/for_tensor_board/initialize_open_gl.jl")

function readFramebufferAsRGBMatrix(windowWidth::Int, windowHeight::Int)
    # Allocate buffer to store pixel data
    pixel_data = Vector{UInt8}(undef, windowWidth * windowHeight * 3)  # 3 channels (RGB)

    # Read pixels from the framebuffer
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, pixel_data)

    # Convert the raw pixel data into a Julia matrix
    rgb_matrix = reshape(pixel_data, (3, windowWidth, windowHeight))
    rgb_matrix = permutedims(rgb_matrix, (3, 2, 1))  # Reorder dimensions to (height, width, channels)

    return rgb_matrix
end

function render_and_save(tf,im_name,step,windowWidth, windowHeight,texture_width,texture_height,dat,line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId)
    xoffset = 0
    yoffset = 0
    glClear(GL_COLOR_BUFFER_BIT)

    # Render the rectangle with texture
    glUseProgram(rectangle_shader_program)

    glBindVertexArray(rectangle_vao[])
    glActiveTexture(textUreId[])
    glBindTexture(GL_TEXTURE_2D, textUreId[])
    glTexSubImage2D(GL_TEXTURE_2D, 0, xoffset, yoffset, texture_width, texture_height, GL_RED, GL_FLOAT, collect(dat))

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
    glBindVertexArray(0)

    # Render the lines
    glUseProgram(line_shader_program)
    glBindVertexArray(line_vao[])
    glDrawElements(GL_LINES, Int(round(length(line_indices) )), GL_UNSIGNED_INT, C_NULL)
    # glDrawElements(GL_LINES, 50, GL_UNSIGNED_INT, C_NULL)
    glBindVertexArray(0)
    GLFW.SwapBuffers(window)
    GLFW.PollEvents()
    rgb_matrix= readFramebufferAsRGBMatrix(windowWidth, windowHeight)
    colorr=permutedims(Float64.(rgb_matrix), (3, 2, 1))
    colorr=colorr./maximum(colorr)
    log_image(tf,im_name,colorr,CWH,step=step)
    return colorr
end



texture_width = 180
texture_height = 180
windowWidth, windowHeight = 800,800
window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program,  line_shader_program, textUreId=initialize_window_etc(windowWidth, windowHeight,texture_width, texture_height)



# Function to generate a random float between -1 and 1
function rand_float_between_neg1_and_1()
    return 2 * rand(Float32) - 1
end

for exp in 1:7
    tf=TBLogger("/home/jakubmitura/projects/MedEye3d.jl/docs/data/hp/exp_$(exp)")
    dummy_hp=Dict("lr"=>rand(0.0001:0.0001:0.1),"batch_size"=>rand(1:10:100),"epochs"=>rand(1:10:100))
    # dummy_metrics=Dict("accuracy"=>rand(0.1:0.1:0.9),"loss"=>rand(0.1:0.1:0.9))
    write_hparams!(tf, dummy_hp, ["accuracy"])
    for step in 1:100
    # Vertex data for lines
        fl=rand_float_between_neg1_and_1()
        line_vertices = Float32[
            fl, 0.0, 0.0,  # top right
            -0.1, 0.0, 0.0,  # bottom right
            0.0, -0.1, 0.0,  # bottom left
            0.0, 0.1, 0.0   # top left
        ]

        # line_vertices = Float32[
        #     0.1, 0.0, 0.0,  # top right
        #     -0.1, 0.0, 0.0,  # bottom right
        #     0.0, -0.1, 0.0,  # bottom left
        #     0.0, 0.1, 0.0   # top left
        # ]


        # Indices for drawing lines
        line_indices = UInt32[
            0, 1,  # Line from top right to bottom right
            2, 3   # Line from bottom left to top left
        ]

        line_vao, line_vbo, line_ebo=initialize_lines(line_vertices, line_indices)


        dat = Float32.(rand(texture_width,texture_height)).*100

        im_name="transverse"
        step=step
        im=render_and_save(tf,im_name,step,windowWidth, windowHeight,texture_width,texture_height,dat,line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId)


        im_name="saggital"
        step=step
        dat = Float32.(rand(texture_width,texture_height)).*100

        im=render_and_save(tf,im_name,step,windowWidth, windowHeight,texture_width,texture_height,dat,line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId)

    end
end
#tensorboard --logdir '/home/jakubmitura/projects/MedEye3d.jl/docs/data/hp'

im
sum(im)

# log_image(tf,"x",im,CWH,step=1)
# log_images(logger::TBLogger, name::AbstractString, imgArrays::AbstractArray, format::ImageFormat; step=step(logger))