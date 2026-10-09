using Vulkan

function patch_file(filepath)
    content = read(filepath, String)
    
    # 1. Patch create_storage_image_3d
    old_code_1 = """
    mem_type_idx = UInt32(0)
    found = false
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
            mem_type_idx = UInt32(i)
            found = true
            break
        end
    end
    if !found
        error("Failed to find suitable memory type for 3D storage image")
    end
    
    alloc_info = MemoryAllocateInfo(mem_reqs.size, mem_type_idx)
    mem = unwrap(allocate_memory(device, alloc_info))"""
    
    new_code_1 = """
    mem_type_idx = UInt32(0)
    mem = nothing
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
            try
                alloc_info = MemoryAllocateInfo(mem_reqs.size, UInt32(i))
                mem = unwrap(allocate_memory(device, alloc_info))
                mem_type_idx = UInt32(i)
                break
            catch
                continue
            end
        end
    end
    if mem === nothing
        error("Failed to allocate $(mem_reqs.size) bytes of DEVICE_LOCAL memory for 3D storage image")
    end"""
    
    content = replace(content, old_code_1 => new_code_1)
    
    # 2. Patch staging box memory
    old_code_2 = """
    mem_type = UInt32(0)
    found_type = false
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            mem_type = UInt32(i)
            found_type = true
            break
        end
    end
    if !found_type
        error("Failed to find HOST_VISIBLE | HOST_COHERENT memory for staging box buffer")
    end
    
    state.staging_box_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, mem_type)))"""
    
    new_code_2 = """
    mem_type = UInt32(0)
    staging_mem = nothing
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            try
                staging_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, UInt32(i))))
                mem_type = UInt32(i)
                break
            catch
                continue
            end
        end
    end
    if staging_mem === nothing
        error("Failed to allocate $(mem_reqs.size) bytes of HOST_VISIBLE | HOST_COHERENT memory for staging box buffer")
    end
    
    state.staging_box_mem = staging_mem"""
    
    content = replace(content, old_code_2 => new_code_2)
    
    # 3. Patch upload_diffusivity!
    old_code_3 = """
    mem_type = UInt32(0)
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            mem_type = UInt32(i)
            break
        end
    end
    alloc_info = MemoryAllocateInfo(mem_reqs.size, mem_type)
    staging_mem = unwrap(allocate_memory(dev, alloc_info))"""
    
    new_code_3 = """
    mem_type = UInt32(0)
    staging_mem = nothing
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            try
                alloc_info = MemoryAllocateInfo(mem_reqs.size, UInt32(i))
                staging_mem = unwrap(allocate_memory(dev, alloc_info))
                mem_type = UInt32(i)
                break
            catch
                continue
            end
        end
    end
    if staging_mem === nothing
        error("Failed to allocate $(mem_reqs.size) bytes of HOST_VISIBLE | HOST_COHERENT memory for staging buffer")
    end"""
    
    content = replace(content, old_code_3 => new_code_3)
    
    write(filepath, content)
    println("Patched!")
end

patch_file("src/display/Vulkan/VulkanHeatDiffusion.jl")
