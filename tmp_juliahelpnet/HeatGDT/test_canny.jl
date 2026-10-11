using Images, ImageEdgeDetection
img = rand(Float32, 64, 64, 64)
try
    edges = canny(img)
    println("Success 3D")
catch e
    println("Error: ", e)
end
