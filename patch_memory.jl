content = read("src/display/Vulkan/VulkanHeatDiffusion.jl", String)

# In init_heat_diffusion!
content = replace(content, """
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            try
                state.staging_box_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, UInt32(i))))
                mem_type = UInt32(i)
                found_type = true
                break
""" => """
    # Prefer HOST_CACHED memory for fast CPU reads
    for pass in 1:2
        target_flags = MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT
        if pass == 1
            target_flags |= MEMORY_PROPERTY_HOST_CACHED_BIT
        end
        
        for i in 0:(mem_props.memory_type_count - 1)
            if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
               (mem_props.memory_types[i + 1].property_flags & target_flags) == target_flags
                try
                    state.staging_box_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, UInt32(i))))
                    mem_type = UInt32(i)
                    found_type = true
                    break
                catch e
                    if e isa Vulkan.VulkanError && e.code == Vulkan.ERROR_OUT_OF_DEVICE_MEMORY
                        continue
                    end
                    rethrow(e)
                end
            end
        end
        if found_type; break; end
    end
    
    if !found_type
        error("Failed to find HOST_VISIBLE | HOST_COHERENT memory or out of memory for staging box buffer")
    end
    
    # We removed the old try-catch because it is inside the loop now
""")

# We need to remove the trailing catch block from the original code since we just replaced the try part but not the catch part!
# Wait, let's just do a more precise replacement using regex or by replacing the whole block.
