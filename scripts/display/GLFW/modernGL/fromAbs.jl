glasserteltype(T)
texture_type = GL_TEXTURE_BUFFER
id = glGenTextures()
glBindTexture(texture_type, id)
internalformat = default_internalcolorformat(T)
glTexBuffer(texture_type, internalformat, buffer.id)
tex = Texture{T, 1}(
    id, texture_type, julia2glenum(T), internalformat,
    default_colorformat(T), TextureParameters(T, 1),
    size(buffer)
)
TextureBuffer(tex, buffer)

gl.LUMINANCE