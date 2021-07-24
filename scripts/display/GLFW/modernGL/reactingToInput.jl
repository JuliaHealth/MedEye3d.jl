####adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3

using Rocket
using GLFW


mutable struct ScrollCallbackSubscribable <: Subscribable{Int}
    xoff_previous :: Float64
    yoff_previous :: Float64
    subject :: Subject{Int}

    ScrollCallbackSubscribable() = new(0.0, 0.0, Subject(Int))
end

function Rocket.on_subscribe!(handler::ScrollCallbackSubscribable, actor)
    return subscribe!(handler.subject, actor)
end

function (handler::ScrollCallbackSubscribable)(_, xoff, yoff)
      # I'm not sure how GLFW represents offsets so it might be wrong here
      # here you actually need to implement your scrolling logic
      if handler.yoff_previous > yoff
          next!(handler.subject, 1)
      else
          next!(handler.subject, -1)
      end 
      handler.xoff_previous = xoff
      handler.yoff_previous = yoff
end

const scrollback = ScrollCallbackSubscribable()

stopListening[]=false


GLFW.SetScrollCallback(window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

# Than later in your application you can do smth like

subscription = subscribe!(scrollback, (direction) -> updateTexture(Int16,widthh,heightt,exampleLabels[200,:,:], trueLabels,stopListening,pboId, DATA_SIZE,GL_UNSIGNED_BYTE)
)
