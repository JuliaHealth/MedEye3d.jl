using ModernGL, GeometryTypes, GLFW



# Create and initialize shaders

#VERTEX Shader
function createVertexShader()
vsh = """
$(get_glsl_version_string())
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;
out vec3 ourColor;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor;
    TexCoord = aTexCoord;
}
"""
return createShader(vsh, GL_VERTEX_SHADER)
end


#FRAGMENT shader
function createFragmentShader()
fsh = """
$(get_glsl_version_string())
out vec4 FragColor;
  
in vec3 ourColor;
in vec2 TexCoord;
uniform sampler2D ourTexture;
void main()
{
    float col=texture(ourTexture, TexCoord).r;   // input color
    FragColor = vec4(col,col,col,1.0f);
}
"""

return createShader(fsh, GL_FRAGMENT_SHADER)
end


