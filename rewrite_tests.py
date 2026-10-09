import re

with open('/mnt/big/project_ssd/project_ssd/MedEye3d.jl/test/test_heatgdt_integration.jl', 'r') as f:
    content = f.read()

replicate_pad_code = """
# Helper: replicate-pad a 3D (N,N,N,1) array by 1 on each side → (N+2,N+2,N+2,1)
function _replicate_pad(arr)
    s = size(arr)
    padded = similar(arr, s[1]+2, s[2]+2, s[3]+2, s[4])
    padded[2:end-1, 2:end-1, 2:end-1, :] .= arr
    padded[1, :, :, :] .= padded[2, :, :, :]
    padded[end, :, :, :] .= padded[end-1, :, :, :]
    padded[:, 1, :, :] .= padded[:, 2, :, :]
    padded[:, end, :, :] .= padded[:, end-1, :, :]
    padded[:, :, 1, :] .= padded[:, :, 2, :]
    padded[:, :, end, :] .= padded[:, :, end-1, :]
    return padded
end
"""

# Insert _replicate_pad before Test 1
content = content.replace("# ─────────────────────────────────────────────────────────────\n# Test 1:", replicate_pad_code + "\n# ─────────────────────────────────────────────────────────────\n# Test 1:")

def replace_D(match):
    var_name = match.group(1) # D or D_4d or D_gpu
    return f"""N = size({var_name}, 1)
        {var_name}_padded = _replicate_pad({var_name})
        D_xp = 0.5f0 .* ({var_name} .+ {var_name}_padded[3:N+2, 2:N+1, 2:N+1, :])
        D_xm = 0.5f0 .* ({var_name} .+ {var_name}_padded[1:N,   2:N+1, 2:N+1, :])
        D_yp = 0.5f0 .* ({var_name} .+ {var_name}_padded[2:N+1, 3:N+2, 2:N+1, :])
        D_ym = 0.5f0 .* ({var_name} .+ {var_name}_padded[2:N+1, 1:N,   2:N+1, :])
        D_zp = 0.5f0 .* ({var_name} .+ {var_name}_padded[2:N+1, 2:N+1, 3:N+2, :])
        D_zm = 0.5f0 .* ({var_name} .+ {var_name}_padded[2:N+1, 2:N+1, 1:N,   :])"""

D_pattern = re.compile(r"D_xp = 0\.5f0 \.\* \(([A-Za-z0-9_]+) \.\+ circshift\(\1, \(-1,\s*0,\s*0,\s*0\)\)\)[\s\S]*?D_zm = 0\.5f0 \.\* \(\1 \.\+ circshift\(\1, \(0,\s*0,\s*1,\s*0\)\)\)")

content = D_pattern.sub(replace_D, content)

def replace_u(match):
    indent = match.group(1)
    return f"""{indent}u_padded = _replicate_pad(u)
{indent}flux = D_xp .* (u_padded[3:N+2, 2:N+1, 2:N+1, :] .- u) .+
{indent}       D_xm .* (u_padded[1:N,   2:N+1, 2:N+1, :] .- u) .+
{indent}       D_yp .* (u_padded[2:N+1, 3:N+2, 2:N+1, :] .- u) .+
{indent}       D_ym .* (u_padded[2:N+1, 1:N,   2:N+1, :] .- u) .+
{indent}       D_zp .* (u_padded[2:N+1, 2:N+1, 3:N+2, :] .- u) .+
{indent}       D_zm .* (u_padded[2:N+1, 2:N+1, 1:N,   :] .- u)"""

u_pattern = re.compile(r"([ \t]*)flux = D_xp \.\* \(circshift\(u, \(-1,\s*0,\s*0,\s*0\)\) \.- u\) \.\+[\s\S]*?D_zm \.\* \(circshift\(u, \(0,\s*0,\s*1,\s*0\)\) \.- u\)")

content = u_pattern.sub(replace_u, content)

# Update test names
content = content.replace("Integration test for the Heat-GDT PDE solver.", "Integration test for the Heat-GDT PDE solver with replicate boundary conditions.")

with open('/mnt/big/project_ssd/project_ssd/MedEye3d.jl/test/test_heatgdt_integration.jl', 'w') as f:
    f.write(content)

print("Rewrite done.")
