using Vulkan
function get_mem(dev, pdev, mem_reqs)
    mem_props = get_physical_device_memory_properties(pdev)
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0
            flags = mem_props.memory_types[i + 1].property_flags
            heap_idx = mem_props.memory_types[i + 1].heap_index
            heap_size = mem_props.memory_heaps[heap_idx + 1].size
            println("Type $i: flags=$flags, heap_size=$(heap_size / 1024^2) MB")
        end
    end
end
