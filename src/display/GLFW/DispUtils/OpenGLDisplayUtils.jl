"""
OpenGLDisplayUtils stub — OpenGL rendering removed, Vulkan backend renders via VulkanRender.
"""
module OpenGLDisplayUtils
export basicRender

function basicRender(window)
    # No-op: Vulkan rendering handled by VulkanRender.render_frame!
end

end #..OpenGLDisplayUtils
