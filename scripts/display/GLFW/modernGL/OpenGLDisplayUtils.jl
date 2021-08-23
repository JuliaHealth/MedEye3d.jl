
module OpenGLDisplayUtils
export  basicRender
using ModernGL
using GLFW

"""
As most functions will deal with just addind the quad to the screen 
and swapping buffers
"""
function basicRender(window)
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
	# Swap front and back buffers
	GLFW.SwapBuffers(window)

end


end #openGLDisplayUtils
