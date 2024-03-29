
import Base: show, similar, Rocket
using Rocket

"""
    AsyncScheduler

`AsyncScheduler` executes scheduled actions asynchronously and uses `Channel` object to order different actions on a single asynchronous task
"""
struct AsyncScheduler{N} <: Rocket.AbstractScheduler end

Base.show(io::IO, ::AsyncScheduler) = print(io, "AsyncScheduler()")

function AsyncScheduler(size::Int = typemax(Int))
    return AsyncScheduler{size}()
end

Base.similar(::AsyncScheduler{N}) where N = AsyncScheduler{N}()

makeinstance(::Type{D}, ::AsyncScheduler{N}) where { D, N } = AsyncSchedulerInstance{D}(N)

instancetype(::Type{D}, ::Type{<:AsyncScheduler}) where D = AsyncSchedulerInstance{D}

struct AsyncSchedulerDataMessage{D}
    data :: D
end

struct AsyncSchedulerErrorMessage
    err
end

struct AsyncSchedulerCompleteMessage end

const AsyncSchedulerMessage{D} = Union{AsyncSchedulerDataMessage{D}, AsyncSchedulerErrorMessage, AsyncSchedulerCompleteMessage}

function identityy(x)
    return x
    end

mutable struct AsyncSchedulerInstance{D}
    channel        :: Channel{AsyncSchedulerMessage{D}}
    isunsubscribed :: Bool
    subscription   :: Rocket.Teardown

    
    AsyncSchedulerInstance{D}(size::Int = typemax(Int)) where D = begin
        chan=Base.Channel{AsyncSchedulerMessage{D}}(identityy,size, spawn=true)
        return new(chan, false, voidTeardown)
    end
end



AsyncSchedulerInstance{Int}(0)

isunsubscribed(instance::AsyncSchedulerInstance) = instance.isunsubscribed
getchannel(instance::AsyncSchedulerInstance) = instance.channel

function dispose(instance::AsyncSchedulerInstance)
    if !isunsubscribed(instance)
        instance.isunsubscribed = true
        close(instance.channel)
        @async begin
            unsubscribe!(instance.subscription)
        end
    end
end

function __process_channeled_message(instance::AsyncSchedulerInstance{D}, message::AsyncSchedulerDataMessage{D}, actor) where D
    on_next!(actor, message.data)
end

function __process_channeled_message(instance::AsyncSchedulerInstance, message::AsyncSchedulerErrorMessage, actor)
    on_error!(actor, message.err)
    dispose(instance)
end

function __process_channeled_message(instance::AsyncSchedulerInstance, message::AsyncSchedulerCompleteMessage, actor)
    on_complete!(actor)
    dispose(instance)
end

struct AsyncSchedulerSubscription{ H <: AsyncSchedulerInstance } <: Teardown
    instance :: H
end

Base.show(io::IO, ::AsyncSchedulerSubscription) = print(io, "AsyncSchedulerSubscription()")

as_teardown(::Type{ <: AsyncSchedulerSubscription}) = UnsubscribableTeardownLogic()

function on_unsubscribe!(subscription::AsyncSchedulerSubscription)
    dispose(subscription.instance)
    return nothing
end

function scheduled_subscription!(source, actor, instance::AsyncSchedulerInstance)
    subscription = AsyncSchedulerSubscription(instance)

    channeling_task = @async begin
        while !isunsubscribed(instance)
            message = take!(getchannel(instance))
            if !isunsubscribed(instance)
                __process_channeled_message(instance, message, actor)
            end
        end
    end

    subscription_task = @async begin
        if !isunsubscribed(instance)
            tmp = on_subscribe!(source, actor, instance)
            if !isunsubscribed(instance)
                subscription.instance.subscription = tmp
            else
                unsubscribe!(tmp)
            end
        end
    end

    bind(getchannel(instance), channeling_task)

    return subscription
end

function scheduled_next!(actor, value::D, instance::AsyncSchedulerInstance{D}) where { D }
    put!(getchannel(instance), AsyncSchedulerDataMessage{D}(value))
end

function scheduled_error!(actor, err, instance::AsyncSchedulerInstance)
    put!(getchannel(instance), AsyncSchedulerErrorMessage(err))
end

function scheduled_complete!(actor, instance::AsyncSchedulerInstance)
    put!(getchannel(instance), AsyncSchedulerCompleteMessage())
end


