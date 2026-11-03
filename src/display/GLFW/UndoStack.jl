"""
    UndoStack

Global undo stack for reversible operations (lesion deletion, etc.).
Ctrl+Z pops the most recent action and invokes its undo closure.
"""
module UndoStack

export UndoRecord, push_undo!, pop_undo!, can_undo, peek_undo, undo_count

"""One undoable action with a description and a closure that restores state."""
struct UndoRecord
    description::String       # e.g. "Delete lesion 3: Right Ilium"
    undo_fn::Function         # () -> nothing; restores state when called
    created_at::Float64       # time() when the action was recorded
end

const _undo_stack = UndoRecord[]
const _undo_lock = ReentrantLock()
const MAX_UNDO = 20

"""Push an undoable record onto the stack (thread-safe)."""
function push_undo!(rec::UndoRecord)
    lock(_undo_lock) do
        push!(_undo_stack, rec)
        while length(_undo_stack) > MAX_UNDO
            popfirst!(_undo_stack)
        end
    end
    @info "[UNDO] Recorded: $(rec.description) (stack depth=$(length(_undo_stack)))"
    nothing
end

"""Pop and execute the most recent undo action. Returns the description or nothing."""
function pop_undo!()::Union{String, Nothing}
    rec = lock(_undo_lock) do
        isempty(_undo_stack) ? nothing : pop!(_undo_stack)
    end
    if rec === nothing
        @info "[UNDO] Nothing to undo."
        return nothing
    end
    @info "[UNDO] Undoing: $(rec.description)"
    try
        rec.undo_fn()
    catch e
        @error "[UNDO] Failed to undo '$(rec.description)': $e" exception=(e, catch_backtrace())
    end
    return rec.description
end

"""Check if there are undoable actions."""
can_undo()::Bool = lock(_undo_lock) do; !isempty(_undo_stack); end

"""Peek at the description of the most recent undoable action."""
function peek_undo()::Union{String, Nothing}
    lock(_undo_lock) do
        isempty(_undo_stack) ? nothing : _undo_stack[end].description
    end
end

"""Number of items on the undo stack."""
undo_count()::Int = lock(_undo_lock) do; length(_undo_stack); end

end # module UndoStack
