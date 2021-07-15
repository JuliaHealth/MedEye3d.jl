using ModernGL, GeometryTypes, GLFW



# Create and initialize shaders

#VERTEX Shader
function createVertexShader()
const vsh = """
$(get_glsl_version_string())
layout (location = 0) in vec3 aPos;
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""
return createShader(vsh, GL_VERTEX_SHADER)
end


#FRAGMENT shader
function createFragmentShader()
const fsh = """
$(get_glsl_version_string())
out vec4 outColor;
void main() {
    outColor = vec4(1.0, 0.5, 1.0, 1.0);
}
"""

return createShader(fsh, GL_FRAGMENT_SHADER)
end


# The shaders 
# vertex_shader = createVertexShader()

# fragment_shader = createFragmentShader()