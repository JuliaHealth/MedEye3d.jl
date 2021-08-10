using DrWatson
@quickactivate "Probabilistic medical segmentation"


"""
The bitmap image of the dot character '.' is much smaller in dimensions than the bitmap image 
https://learnopengl.com/In-Practice/Text-Rendering

"""


using DrWatson
@quickactivate "Probabilistic medical segmentation"

using FreeTypeAbstraction

using Main.PrepareWindowHelpers
using Main.OpenGLDisplayUtils
using Main.TextureManag
using Main.ShadersAndVerticiesForText
using Glutils
using FreeTypeAbstraction


################ basics
window = initializeWindow(1000,600)
println(createcontextinfo())

gslsStr = get_glsl_version_string()

vertex_shaderB = ShadersAndVerticiesForText.createVertexShader(gslsStr)
fragment_shaderB = ShadersAndVerticiesForText.createFragmentShader(gslsStr)

shader_program = glCreateProgram()


glAttachShader(shader_program, vertex_shaderB)
glAttachShader(shader_program, fragment_shaderB)
glLinkProgram(shader_program)
glUseProgram(shader_program)


    

############### end basics

texture = TextureManag.createTexture(1,Int32(200),Int32(800),GL_R8I)

glActiveTexture(GL_TEXTURE0 +8); # active proper texture unit before binding
glBindTexture(GL_TEXTURE_2D, texture[]); 

samplerRefNumb= glGetUniformLocation(shader_program, "TextTexture1")
glUniform1i(samplerRefNumb,1);# we first look for uniform sampler in shader  





# vbo = createDAtaBuffer(Main.ShadersAndVerticiesForText.verticesB)
# # Create the Element Buffer Object (EBO)
# ebo = createElementBuffer(Main.ShadersAndVerticiesForText.elements)
# ############ how data should be read from data buffer
# encodeDataFromDataBuffer()



face = FreeTypeAbstraction.findfont("hack";  additional_fonts= datadir("fonts"))
img, extent = renderface(face, 'C', 64)

# render a string into an existing matrix
a = renderstring!(
    zeros(UInt8, 40, 40),
    "uuuuuuu1111",
    face,
    5,
    5,
    5,
    valign = :vbottom,
)




basicRender(window)


vbo = createDAtaBuffer(Main.ShadersAndVerticies.vertices)
	# Create the Element Buffer Object (EBO)
	ebo = createElementBuffer(Main.ShadersAndVerticies.elements)
	############ how data should be read from data buffer
	encodeDataFromDataBuffer()