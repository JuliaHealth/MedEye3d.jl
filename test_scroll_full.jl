using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
m1 = Menu(g[1,1], options=["A"])
m2 = Menu(g[2,1], options=["B"])

_scroll_offset_px = Ref(0.0f0)
_bbox_cache = Dict{Int, Any}()
_bbox_cache_ready = Ref(false)

function _cache_widget_bboxes!()
    empty!(_bbox_cache)
    for (i, c) in enumerate(g.content)
        w = c.content
        bb = w.layoutobservables.computedbbox[]
        _bbox_cache[i] = bb
    end
    _bbox_cache_ready[] = true
end

function _apply_scroll!(offset_px::Float32)
    _scroll_offset_px[] = offset_px
    !_bbox_cache_ready[] && return
    for (i, c) in enumerate(g.content)
        w = c.content
        orig = get(_bbox_cache, i, nothing)
        orig === nothing && continue
        w.layoutobservables.computedbbox[] = Rect2f(orig.origin[1], orig.origin[2] + offset_px, orig.widths[1], orig.widths[2])
    end
end

# 1. Initial layout & cache
Makie.GridLayoutBase.update!(g)
_cache_widget_bboxes!()

# 2. User scrolls down by 200px
_apply_scroll!(200.0f0)
println("After scroll 200px: m1 y = ", m1.layoutobservables.computedbbox[].origin[2])
println("After scroll 200px: m2 y = ", m2.layoutobservables.computedbbox[].origin[2])

# 3. User interacts with anatomy UI (e.g. add/remove row)
# update_anatomy_ui executes:
g.block_updates = false
try
    rowsize!(g, 2, 50)
    Makie.GridLayoutBase.update!(g)
finally
    g.block_updates = true
    if _bbox_cache_ready[]
        _cache_widget_bboxes!()
        _apply_scroll!(_scroll_offset_px[])
    end
end

println("After interaction & restore: m1 y = ", m1.layoutobservables.computedbbox[].origin[2])
println("After interaction & restore: m2 y = ", m2.layoutobservables.computedbbox[].origin[2])

# 4. Another interaction
g.block_updates = false
try
    rowsize!(g, 2, 30)
    Makie.GridLayoutBase.update!(g)
finally
    g.block_updates = true
    if _bbox_cache_ready[]
        _cache_widget_bboxes!()
        _apply_scroll!(_scroll_offset_px[])
    end
end

println("After 2nd interaction & restore: m1 y = ", m1.layoutobservables.computedbbox[].origin[2])
println("After 2nd interaction & restore: m2 y = ", m2.layoutobservables.computedbbox[].origin[2])

