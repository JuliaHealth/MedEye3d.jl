using ModernGL, GeometryTypes, GLFW


# Now we define another geometry that we will render, a rectangle, this one with an index buffer
# The positions of the vertices in our rectangle
positions = Point{2,Float32}[(-0.5,  0.5),     # top-left
                             ( 0.5,  0.5),     # top-right
                             ( 0.5, -0.5),     # bottom-right
                             (-0.5, -0.5)]     # bottom-left

# Specify how vertices are arranged into faces
# Face{N,T} type specifies a face with N vertices, with index type
# T (you should choose UInt32), and index-offset O. If you're
# specifying faces in terms of julia's 1-based indexing, you should set
# O=0. (If you instead number the vertices starting with 0, set
# O=-1.)
elements = Face{3,UInt32}[(0,1,2),          # the first triangle
                          (2,3,0)]          # the second triangle



# texture coordinates

positions = Point{2,Float32}[(-0.5,  0.5),     # top-left
                          ( 0.5,  0.5),     # top-right
                          ( 0.5, -0.5),     # bottom-right
                          (-0.5, -0.5)]     # bottom-left


 vertices = Float32.([
                            # positions          // colors           // texture coords
                             0.5,  0.5, 0.0,   1.0, 0.0, 0.0,   1.0, 1.0,   # top right
                             0.5, -0.5, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,   # bottom right
                            -0.5, -0.5, 0.0,   0.0, 0.0, 1.0,   0.0, 0.0,   # bottom left
                            -0.5,  0.5, 0.0,   1.0, 1.0, 0.0,   0.0, 1.0    # top left 
 ])
