
module OpenGLDisplayUtils
export  basicRender

basicRenderDoc = """
As most functions will deal with just addind the quad to the screen 
and swapping buffers
"""
@doc basicRenderDoc
function basicRender()
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
	# Swap front and back buffers
	GLFW.SwapBuffers(window)

end


end #openGLDisplayUtils