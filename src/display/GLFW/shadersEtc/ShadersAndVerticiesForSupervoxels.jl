"""
ShadersAndVerticiesForSupervoxels stub — OpenGL supervoxel rendering removed, Vulkan backend handles this.
"""
module ShadersAndVerticiesForSupervoxels
using GLFW

export createAndInitSupervoxelLineShaderProgram, renderSupervoxelLines

function createAndInitSupervoxelLineShaderProgram(vertex_shader)
    return (UInt32(0), UInt32(0))
end

function renderSupervoxelLines(forDispObj, supervoxelFields, mainRectFields, allSupervoxels, dim, current)
    # No-op: supervoxel rendering not yet ported to Vulkan
end

end # module ShadersAndVerticiesForSupervoxels
