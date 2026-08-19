vol = zeros(5, 5)
slice = vol[2:4, 2:4]
slice[1, 1] = 1.0
println(sum(vol))
