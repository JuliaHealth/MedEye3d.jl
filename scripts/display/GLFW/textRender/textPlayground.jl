using DrWatson
@quickactivate "Probabilistic medical segmentation"

using FreeType

function getLibrary()
    library = Ref{FT_Library}()
    error = FT_Init_FreeType(library)
    return library
end


function getFace(library)
    refface = Ref{FT_Face}()
    face = FT_New_Face(library[], DrWatson.datadir("fonts","hack_regular.ttf"), 0, refface) 
    return refface
end 



CharacterStructStr = """
adapted from 
https://learnopengl.com/In-Practice/Text-Rendering
"""
@doc CharacterStructStr
struct CharacterStruct 
    TextureID  # ID handle of the glyph texture
    Size       # Size of glyph
    Bearing    # Offset from baseline to left/top of glyph
    Advance    # Offset to advance to next glyph
end#Character




library = getLibrary()
face = getFace(library)
FT_Set_Pixel_Sizes(face[], 0, 48);  
FT_Load_Char(face[], 'X', FT_LOAD_RENDER)


"""
adapted from https://learnopengl.com/In-Practice/Text-Rendering
we will store data  about fonts in textures 
"""

function loadFonts()
glPixelStorei(GL_UNPACK_ALIGNMENT, 1); # disable byte-alignment restriction

for i = 0:128
    # load character glyph 
    FT_Load_Char(face, c, FT_LOAD_RENDER)

    # generate texture
    texture= Ref(GLuint(numb));
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, texture[]); 

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        face->glyph->bitmap.width,
        face->glyph->bitmap.rows,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        face->glyph->bitmap.buffer
    );
    # set texture options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    # now store character for later use
    Character character = {
        texture, 
        glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
        glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
        face->glyph->advance.x
    };
    Characters.insert(std::pair<char, Character>(c, character));
end#for

end #loadFonts




"""
The bitmap image of the dot character '.' is much smaller in dimensions than the bitmap image 
https://learnopengl.com/In-Practice/Text-Rendering

"""


using DrWatson
@quickactivate "Probabilistic medical segmentation"

using FreeTypeAbstraction


face = FreeTypeAbstraction.findfont("hack";  additional_fonts= datadir("fonts"))
img, extent = renderface(face, 'C', 64)



# render a string into an existing matrix
a = renderstring!(
    zeros(UInt8, 40, 40),
    "ilililililil",
    face,
    5,
    5,
    5,
    valign = :vbottom,
)
@test any(a[1:10, :] .!= 0)
@test all(a[11:20, :] .== 0)
maximum(a)
minimum(a)


####### creating second quad


elements = Face{3,UInt32}[(0,1,2),          # the first triangle
(2,3,0)]          # the second triangle



verticesB = Float32.([
  # positions          // colors           // texture coords
   1.0,  1.0, 0.0,   1.0, 0.0, 0.0,   1.0, 1.0,   # top right
   1.0, -1.0, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,   # bottom right
   0.8, -1.0, 0.0,   0.0, 0.0, 1.0,   0.0, 0.0,   # bottom left
   0.8,  1.0, 0.0,   1.0, 1.0, 0.0,   0.0, 1.0    # top left 
   ])




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


using Main.PrepareWindowHelpers
using Main.OpenGLDisplayUtils
using Main.TextureManag
using Main.ShadersAndVerticiesForText
using Glutils
using FreeTypeAbstraction


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