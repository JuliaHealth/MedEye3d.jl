content = read("src/display/LesionMetadataWindow.jl", String)

old_code = """
                    Makie.GridLayoutBase.update!(g)
                finally
                    g.block_updates = was_blocked
                    if _bbox_cache_ready[]
                        _cache_widget_bboxes!()
                        _apply_scroll!(_scroll_offset_px[])
                    end
                end
"""

new_code = """
                    Makie.GridLayoutBase.update!(g)
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
