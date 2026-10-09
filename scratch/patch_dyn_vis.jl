content = read("src/display/LesionMetadataWindow.jl", String)

old_code = """
        finally
            # g.block_updates = false
            # try Makie.GridLayoutBase.update!(g) catch; end
        end
"""

new_code = """
        finally
            @async begin
                sleep(0.15)
                _cache_widget_bboxes!()
                g.block_updates = true
                _apply_scroll!(_scroll_offset_px[])
            end
        end
"""
content = replace(content, old_code => new_code)
write("src/display/LesionMetadataWindow.jl", content)
