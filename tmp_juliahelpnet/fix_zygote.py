import re

with open('full_train.jl', 'r') as f:
    content = f.read()

# Replace the Zygote block
old_block = """            loss, gs, st_new, ce, dl = Zygote.withgradient(ps) do p
                loss, st_n, c, d = compute_loss(model, p, st, x, y)
                loss
            end"""
new_block = """            (loss, st_new, ce, dl), back = Zygote.pullback(p -> compute_loss(model, p, st, x, y), ps)
            gs = back((1.0f0, nothing, nothing, nothing))[1]"""

content = content.replace(old_block, new_block)

with open('full_train.jl', 'w') as f:
    f.write(content)
