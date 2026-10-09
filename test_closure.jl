function test()
    dim = 1
    z = 10
    pts = [CartesianIndex(10, 2, 3), CartesianIndex(11, 2, 3)]
    get_slice_set = (pts) -> begin
        slice_pts = filter(p -> (dim == 1 ? p[1] : (dim == 2 ? p[2] : p[3])) == z, pts)
        return Set(slice_pts)
    end
    s = get_slice_set(pts)
    println(s)
end
test()
