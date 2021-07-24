using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ReactingToInput
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
      if handler.yoff_previous > yoff
          next!(handler.subject, 1)
      else
          next!(handler.subject, -1)
      end 
      handler.xoff_previous = xoff
      handler.yoff_previous = yoff
end


registerMouseScrollFunctionsStr="""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end 
"""
@doc registerMouseScrollFunctionsStr
function registerMouseScrollFunctions()

scrollback = ScrollCallbackSubscribable()
stopListening[]=true # stoping event listening loop to free the GLFW context

GLFW.SetScrollCallback(window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

subscription = subscribe!(scrollback, (direction) -> println(direction))


stopListening[]=false # reactivate event listening loop

end #registerMouseScrollFunctions

end #ReactToGLFWInpuut