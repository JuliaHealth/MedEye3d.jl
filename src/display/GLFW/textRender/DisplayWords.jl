"""
DisplayWords stub — OpenGL text rendering removed.
Text display functionality is no longer needed with the Vulkan backend.
Functions are kept as stubs for API compatibility with callers.
"""
module DisplayWords
using FreeTypeAbstraction, ColorTypes
using ..ForDisplayStructs, ..DataStructs

export getTextForCurrentSlice, textLinesFromStrings, renderSingleLineOfText
export activateForTextDisp, bindAndActivateForText, reactivateMainObj
export createTextureForWords, bindAndDisplayTexture, addTextToTexture


"""
Get text lines for current slice (pure CPU function, no OpenGL).
"""
function getTextForCurrentSlice(onScrollData, sliceNum::Int32)
    return SimpleLineTextStruct[]
end

"""
Create text lines from strings (pure CPU, no OpenGL).
"""
function textLinesFromStrings(strs::Vector{String})
    return map(s -> SimpleLineTextStruct(text=s), strs)
end

function renderSingleLineOfText(args...)
    # No-op: text rendering removed
    return nothing
end

function activateForTextDisp(shader_program_words, vbo_words, calcDim)
    # No-op: text display removed
end

function bindAndActivateForText(args...)
    # No-op: text display removed
end

function reactivateMainObj(shader_program, vbo, calcDim)
    # No-op: OpenGL buffer reactivation removed, Vulkan handles this
end

function createTextureForWords(numberOfActiveTextUnits, widthh, heightt, glTexture)
    # Return dummy TextureSpec for text (won't be used)
    return TextureSpec()
end

function bindAndDisplayTexture(args...)
    # No-op: text display removed
end

function addTextToTexture(wordsDispObj, lines, calcDimStruct)
    # No-op: text rendering removed
    return nothing
end

end # module DisplayWords
