"""
ShadersAndVerticiesForText stub — OpenGL text shaders removed, text rendering not needed for Vulkan backend.
"""
module ShadersAndVerticiesForText


export getWordsVerticies, createFragmentShader

function getWordsVerticies(fractionOfMainIm)
    return Float32[0,0,0,0,0,0,0,0]
end

function createFragmentShader(gslsStr)
    return UInt32(0)
end

end # module ShadersAndVerticiesForText
