t1 = @elapsed zeros(Float32, 1024, 1024, 326)
println("zeros(Float32) [1024x1024x326]: $(round(t1*1000, digits=1)) ms")
