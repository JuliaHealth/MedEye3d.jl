
module OpenGLDisplayUtils
export basicRender
using ModernGL
using GLFW, Base.Threads
# using ..ShadersAndVerticiesForLine

"""
As most functions will deal with just addind the quad to the screen
and swapping buffers

Incorporating rendering for crosshair
"""
# function basicRender(window)
#     glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
# end

function basicRender(window)
    # We disable individual render and swap here, because we want to render all panels
    # and swap once at the end of the event loop in the consumer.
end
end #..OpenGLDisplayUtils
