# test_pde_adjoint.jl — Verify analytic adjoint correctness via finite differences
# Tests gradients for ∂L/∂D, ∂L/∂β, ∂L/∂v against finite differences

using Pkg; Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using CUDA
using LuxCUDA
using Zygote
using Statistics
using Printf

include("heat_diffuse.jl")
include("heat_diffuse_v2.jl")

using .HeatDiffusion
using .HeatDiffusionV2

function test_adjoint()
    println("="^60)
    println("PDE Adjoint Correctness Test (Finite Differences)")
    println("="^60)

    # Small test volume (8×8×8)
    N = 8
    B = 1
    K = 5
    dt = 0.05f0
    eps_fd = 2f-3  # 0.002: optimal balance between Float32 ULP and O(eps^2) truncation error

    # Create test inputs on GPU with positive field so gradients are non-vanishing
    u0 = CUDA.rand(Float32, N, N, N, B) .* 0.5f0 .+ 0.5f0
    D = CUDA.rand(Float32, N, N, N, B) .* 0.5f0 .+ 0.2f0
    beta = CUDA.rand(Float32, N, N, N, B) .* 0.2f0 .+ 0.05f0
    v = CUDA.rand(Float32, N, N, N, 3, B) .* 0.1f0 .- 0.05f0

    # Normalized loss: mean(u .^ 2) so values stay around O(1) where Float32 has high precision
    function loss_fn(u0, D, beta, v)
        u = HeatDiffusionV2.heat_diffusion_v2(u0, D, beta, v, K, dt)
        return mean(u .^ 2)
    end

    # ── Get analytic gradients via Zygote ──
    println("\nComputing analytic gradients via Zygote + custom rrule...")
    L, (g_u0, g_D, g_beta, g_v) = Zygote.withgradient(loss_fn, u0, D, beta, v)
    println("  Loss value: $L")

    all_passed = true

    # ── Test ∂L/∂D via finite differences ──
    println("\n1. Testing ∂L/∂D...")
    D_cpu = Array(D)
    g_D_cpu = Array(g_D)
    n_test = 5
    max_err = 0.0
    for trial in 1:n_test
        i, j, k = rand(2:N-1), rand(2:N-1), rand(2:N-1)
        D_plus = copy(D_cpu); D_plus[i, j, k, 1] += eps_fd
        D_minus = copy(D_cpu); D_minus[i, j, k, 1] -= eps_fd
        L_plus = loss_fn(u0, CuArray(D_plus), beta, v)
        L_minus = loss_fn(u0, CuArray(D_minus), beta, v)
        fd_grad = (L_plus - L_minus) / (2f0 * eps_fd)
        analytic = g_D_cpu[i, j, k, 1]
        rel_err = abs(fd_grad - analytic) / max(abs(analytic), 1f-4)
        max_err = max(max_err, rel_err)
        @printf("  [%d,%d,%d] Analytic=%.6f FD=%.6f RelErr=%.3f%%\n",
                i, j, k, analytic, fd_grad, rel_err * 100)
    end
    passed = max_err < 0.05  # 5% tolerance in Float32
    println("  ∂L/∂D: $(passed ? "✓ PASS" : "✗ FAIL") (max rel err: $(round(max_err*100, digits=3))%)")
    all_passed &= passed

    # ── Test ∂L/∂β via finite differences ──
    println("\n2. Testing ∂L/∂β...")
    beta_cpu = Array(beta)
    g_beta_cpu = Array(g_beta)
    max_err = 0.0
    for trial in 1:n_test
        i, j, k = rand(2:N-1), rand(2:N-1), rand(2:N-1)
        b_plus = copy(beta_cpu); b_plus[i, j, k, 1] += eps_fd
        b_minus = copy(beta_cpu); b_minus[i, j, k, 1] -= eps_fd
        L_plus = loss_fn(u0, D, CuArray(b_plus), v)
        L_minus = loss_fn(u0, D, CuArray(b_minus), v)
        fd_grad = (L_plus - L_minus) / (2f0 * eps_fd)
        analytic = g_beta_cpu[i, j, k, 1]
        rel_err = abs(fd_grad - analytic) / max(abs(analytic), 1f-4)
        max_err = max(max_err, rel_err)
        @printf("  [%d,%d,%d] Analytic=%.6f FD=%.6f RelErr=%.3f%%\n",
                i, j, k, analytic, fd_grad, rel_err * 100)
    end
    passed = max_err < 0.05
    println("  ∂L/∂β: $(passed ? "✓ PASS" : "✗ FAIL") (max rel err: $(round(max_err*100, digits=3))%)")
    all_passed &= passed

    # ── Test ∂L/∂v via finite differences ──
    println("\n3. Testing ∂L/∂v...")
    v_cpu = Array(v)
    g_v_cpu = Array(g_v)
    max_err = 0.0
    for trial in 1:n_test
        i, j, k = rand(2:N-1), rand(2:N-1), rand(2:N-1)
        d = rand(1:3)  # direction component
        v_plus = copy(v_cpu); v_plus[i, j, k, d, 1] += eps_fd
        v_minus = copy(v_cpu); v_minus[i, j, k, d, 1] -= eps_fd
        L_plus = loss_fn(u0, D, beta, CuArray(v_plus))
        L_minus = loss_fn(u0, D, beta, CuArray(v_minus))
        fd_grad = (L_plus - L_minus) / (2f0 * eps_fd)
        analytic = g_v_cpu[i, j, k, d, 1]
        rel_err = abs(fd_grad - analytic) / max(abs(analytic), 1f-4)
        max_err = max(max_err, rel_err)
        @printf("  [%d,%d,%d,%d] Analytic=%.6f FD=%.6f RelErr=%.3f%%\n",
                i, j, k, d, analytic, fd_grad, rel_err * 100)
    end
    passed = max_err < 0.05
    println("  ∂L/∂v: $(passed ? "✓ PASS" : "✗ FAIL") (max rel err: $(round(max_err*100, digits=3))%)")
    all_passed &= passed

    # ── Test backward-compatible wrapper (β=0, v=0) ──
    println("\n4. Testing backward-compatible wrapper (β=0, v=0)...")
    function loss_compat(u0, D)
        u = HeatDiffusionV2.heat_diffusion_v2(u0, D, K, dt)
        return mean(u .^ 2)
    end
    L_compat, (g_u0_c, g_D_c) = Zygote.withgradient(loss_compat, u0, D)
    println("  V2 wrapper loss: $L_compat")

    # Compare with original heat_diffuse.jl
    function loss_orig(u0, D)
        u = HeatDiffusion.heat_diffusion(u0, D, K, dt)
        return mean(u .^ 2)
    end
    L_orig, (g_u0_o, g_D_o) = Zygote.withgradient(loss_orig, u0, D)
    println("  Original solver loss: $L_orig")
    diff = abs(L_compat - L_orig) / (abs(L_orig) + 1f-8)
    println("  Forward loss difference: $(round(diff * 100, digits=6))%")
    
    # Check gradient match between V2 wrapper and original
    g_D_diff = mean(abs.(Array(g_D_c) .- Array(g_D_o))) / (mean(abs.(Array(g_D_o))) + 1f-8)
    println("  Gradient ∂L/∂D match: $(round((1 - g_D_diff)*100, digits=4))%")

    passed = diff < 1e-4 && g_D_diff < 1e-4
    println("  Backward compat: $(passed ? "✓ PASS" : "✗ FAIL")")
    all_passed &= passed

    println("\n" * "="^60)
    println(all_passed ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗")
    println("="^60)
end

test_adjoint()
