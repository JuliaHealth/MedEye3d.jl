#copied from https://github.com/tkf/Awaits.jl/blob/master/src/Awaits.jl

module Awaits

# Use README as the docstring of the module:

export @taskgroup, @cancelscope, @in, @go, @await, @check, cancel!

using MacroTools: postwalk, @capture

struct Cancelled <: Exception end

function locking(f, l)
    lock(l)
    try
        return f()
    finally
        unlock(l)
    end
end

"""
    SyncSeq([values])
A minimalistic wrapper of `Vector{Any}` that can be used from multiple
threads.
"""
struct SyncSeq
    values::Vector{Any}
    lock::Threads.SpinLock
end

SyncSeq() = SyncSeq([])
SyncSeq(values::Vector{Any}) = SyncSeq(values, Threads.SpinLock())

Base.push!(v::SyncSeq, x) = locking(v.lock) do
    push!(v.values, x)
end

Base.iterate(v::SyncSeq, i=1) = locking(v.lock) do
    iterate(v.values, i)
end

# For `collect` in test code:
Base.IteratorSize(::Type{<:SyncSeq}) = Base.SizeUnknown()

struct TaskContext
    tasks::SyncSeq
    listening::Vector{Threads.Atomic{Bool}}
    cancelables::Vector{Threads.Atomic{Bool}}
end

function TaskContext()
    c = Threads.Atomic{Bool}(false)
    return TaskContext(SyncSeq(), [c], [c])
end

function newcontext!(ctx::TaskContext)
    c = Threads.Atomic{Bool}(false)
    push!(ctx.cancelables, c)

    return TaskContext(ctx.tasks, push!(copy(ctx.listening), c), [c])
end

function shouldstop(ctx::TaskContext)
    for c in ctx.listening
        c[] && return true
    end
    return false
end

"""
    cancel!(context::TaskContext)
Cancel (stop) execution of task `context`.
"""
function cancel!(ctx::TaskContext)
    for c in ctx.cancelables
        c[] = true
    end
end

_copy(ctx::TaskContext) =
    TaskContext(ctx.tasks, copy(ctx.listening), copy(ctx.cancelables))

_tasks(ctx::TaskContext) =
    (begin
         shouldstop(ctx) && cancel!(ctx)
         task
     end
     for task in ctx.tasks)

const ctx_varname = gensym("taskcontext")

"""
    @taskgroup(body)
Create a new task context.  This is an extension of the `@sync` block.
"""
macro taskgroup(body)
    ctx_var = esc(ctx_varname)
    sync_var = esc(Base.sync_varname)
    quote
        # TODO: "inherit" old task context when a new context is created
        let $ctx_var = TaskContext(),
            $sync_var = $ctx_var.tasks

            v = $(esc(body))
            Base.sync_end(_tasks($ctx_var))
            v
        end
    end
end

"""
    @in context body
Enter into the task `context` so that other macros like `@go` works as
expected.  This should be used inside a function in which the task
`context` is passed by a caller (for example, using [`@go f(_)`](@ref
@go) macro).
"""
macro in(ctx, body)
    ctx_var = esc(ctx_varname)
    sync_var = esc(Base.sync_varname)
    quote
        let $ctx_var = $(esc(ctx)),
            $sync_var = $ctx_var.tasks

            $(esc(body))
        end
    end
end

"""
    @cancelscope(body) :: TaskContext
Create a task context with a new scope of cancellation.  Tasks
launched inside a `@cancelscope` can be cancelled independently from
the tasks outside to this scope.  This scope returns a context object
that can be used to cancel the task by calling [`cancel!`](@ref) on
this object.
"""
macro cancelscope(body)
    ctx_var = esc(ctx_varname)
    quote
        let $ctx_var = newcontext!($ctx_var)
            $(esc(body))
            $ctx_var
        end
    end
end

substitute_context(ctx, body) =
    postwalk(body) do ex
        if @capture(ex, f_(xs__)) && :_ in xs
            :($f($((x == :_ ? ctx : x for x in xs)...)))
        else
            ex
        end
    end

"""
    @await body
Handle an exception returned from `body`.  If `body` evaluates to an
`Exception`, it cancels all the tasks attached to the current context.
The variable `_` in code like `@await f(_, x, y)` is replaced by the
current task context.  It must be invoked inside `@taskgroup` or `@in`
macros.
"""
macro await(body)
    quote
        ans = $(esc(substitute_context(ctx_varname, body)))
        if ans isa Exception
            cancel!($(esc(ctx_varname)))
            return ans
        end
        ans
    end
end

"""
    @go body
A wrapper of `Threads.@spawn` that makes cancellation work.
The variable `_` in code like `@go f(_, x, y)` is replaced by the
current task context.  It must be invoked inside `@taskgroup` or `@in`
macros.
"""
macro go(body)
    quote
        # Copy task context to avoid touching `Vector` without a lock.
        # It means that the `@spawn`ed task will miss future additions
        # of cancellation tokens in this task.  However, since current
        # task context and the copied task context share at least one
        # cancellation token (the root one), the cancellation state
        # will propagate eventually.
        let $ctx_varname = $_copy($ctx_varname)
            $Threads.@spawn $(@__MODULE__).@await begin
                $(substitute_context(ctx_varname, body))
            end
        end
    end |> esc
end

"""
    @check [context]
Check `context` and exit the function (do `return Cancelled()`) if it
is required.
"""
macro check()
    quote
        @check($ctx_varname)
    end |> esc
end

macro check(ctx)
    ctx_var = esc(ctx)
    quote
        shouldstop($ctx_var) && return Cancelled()
    end
end

end # module
