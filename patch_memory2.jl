content = read("src/display/Vulkan/VulkanHeatDiffusion.jl", String)

old_block = """
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            try
                state.staging_box_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, UInt32(i))))
                mem_type = UInt32(i)
                found_type = true
                break
            catch e
                if e isa Vulkan.VulkanError && e.code == Vulkan.ERROR_OUT_OF_DEVICE_MEMORY
                    @warn "Staging buffer memory allocation failed on heap \$(i), trying next suitable heap..."
                    continue
                end
                rethrow(e)
            end
        end
    end
"""

new_block = """
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
                        @warn "Staging buffer memory allocation failed on heap \$(i), trying next suitable heap..."
                        continue
                    end
                    rethrow(e)
                end
            end
        end
        if found_type; break; end
    end
"""

content = replace(content, old_block => new_block)
write("src/display/Vulkan/VulkanHeatDiffusion.jl", content)
