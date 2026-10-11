# losses.jl — Loss functions for Heat-GDT framework
# Included directly (no module wrapper)

using Statistics: mean, std
using NNlib

"""
    compute_3d_dtm(gt_mask)

Compute 3D Distance Transform Map from ground truth binary mask.
`gt_mask`: `(X, Y, Z, B)` Float32 with 0/1 values.
Returns: `(X, Y, Z, B)` Float32 distance map.

Uses a fast approximation: iterative min-pooling with stride=1, pad=1 using
`NNlib.maxpool` on negated mask, running multiple iterations to propagate distances.
"""
function compute_3d_dtm(gt_mask)
    X, Y, Z, B = size(gt_mask)
    gt5 = reshape(gt_mask, X, Y, Z, 1, B)

    # Compute boundary: surface voxels (foreground voxels adjacent to background)
    bg = 1.0f0 .- gt5
    boundary = gt5 .* NNlib.maxpool(bg, (3, 3, 3); stride=1, pad=1)

    # Initialize distance: 0 at boundary, large elsewhere
    dist = reshape(boundary, X, Y, Z, B)
    # Approximate chamfer distance via iterative dilation
    d = (1.0f0 .- dist) .* 10.0f0  # 0 at boundary, 10 elsewhere
    for iter in 1:5
        d5 = reshape(d, X, Y, Z, 1, B)
        # Min-pool = negative of max-pool of negative
        d_min = -NNlib.maxpool(-d5, (3, 3, 3); stride=1, pad=1)
        d = reshape(min.(d5, d_min .+ 1.0f0), X, Y, Z, B)
    end
    return d
end

using Zygote

"""
    boundary_loss_dtm(pred_mask, gt_mask)

Kervadec et al. Boundary Loss using Distance Transform Maps.
Penalizes predictions proportionally to their distance from the true boundary.
"""
function boundary_loss_dtm(pred_mask, gt_mask)
    signed_dtm = Zygote.ignore_derivatives() do
        dtm = compute_3d_dtm(gt_mask)
        dtm .* (1.0f0 .- 2.0f0 .* gt_mask)
    end
    return mean(signed_dtm .* pred_mask)
end

"""
    focal_tversky_loss(pred_mask, gt_mask; alpha=0.3f0, beta_tv=0.7f0, gamma_ft=0.75f0)

Focal Tversky Loss that heavily penalizes false negatives on boundary voxels.

# Keyword Arguments
- `alpha`: weight for false positives (default `0.3f0`)
- `beta_tv`: weight for false negatives (default `0.7f0`)
- `gamma_ft`: focal exponent (default `0.75f0`)
"""
function focal_tversky_loss(pred_mask, gt_mask; alpha=0.3f0, beta_tv=0.7f0, gamma_ft=0.75f0)
    p = clamp.(pred_mask, 1.0f-6, 1.0f0 - 1.0f-6)
    tp = sum(p .* gt_mask, dims=(1, 2, 3))
    fp = sum(p .* (1.0f0 .- gt_mask), dims=(1, 2, 3))
    fn = sum((1.0f0 .- p) .* gt_mask, dims=(1, 2, 3))
    ti = (tp .+ 1.0f-5) ./ (tp .+ alpha .* fp .+ beta_tv .* fn .+ 1.0f-5)
    return mean((1.0f0 .- ti) .^ gamma_ft)
end

"""
    dice_loss(pred_mask, gt_mask)

Soft Dice loss over spatial dimensions `(1, 2, 3)`, averaged across batch.
"""
function dice_loss(pred_mask, gt_mask)
    intersection = sum(pred_mask .* gt_mask, dims=(1, 2, 3))
    union_sum = sum(pred_mask, dims=(1, 2, 3)) .+ sum(gt_mask, dims=(1, 2, 3))
    dsc = (2.0f0 .* intersection .+ 1.0f-5) ./ (union_sum .+ 1.0f-5)
    return mean(1.0f0 .- dsc)
end

"""
    bce_loss(pred_mask, gt_mask)

Element-wise binary cross-entropy loss with numerical clamping, averaged over all elements.
"""
function bce_loss(pred_mask, gt_mask)
    p = clamp.(pred_mask, 1.0f-6, 1.0f0 - 1.0f-6)
    loss = -(gt_mask .* log.(p) .+ (1.0f0 .- gt_mask) .* log.(1.0f0 .- p))
    return mean(loss)
end

"""
    compound_loss(pred_mask, edge_map, gt_mask; w_dice=1.0f0, w_bce=0.5f0, w_dtm=0.4f0, w_sparse=0.01f0)

Full compound loss with DTM boundary supervision.

Returns a tuple `(total, dice_l, bce_l, dtm_l, sparse_l)`.
"""
function compound_loss(pred_mask, edge_map, gt_mask;
                       w_dice=1.0f0, w_bce=0.5f0, w_dtm=0.4f0, w_sparse=0.01f0)
    dl    = dice_loss(pred_mask, gt_mask)
    bl    = bce_loss(pred_mask, gt_mask)
    dtm_l = boundary_loss_dtm(pred_mask, gt_mask)
    sp    = mean(edge_map)
    total = w_dice * dl + w_bce * bl + w_dtm * dtm_l + w_sparse * sp
    return total, dl, bl, dtm_l, sp
end

"""
    edge_supervision_loss(pred_edge, edge_gt; pos_weight=3.0f0)

Combined weighted BCE + Dice loss for edge prediction vs TotalSegmentator edge ground truth.
Uses positive class weighting since organ boundary edges are ~25% of voxels.

For samples WITHOUT edge GT (all zeros), this reduces to -mean(log(1-p)) which
encourages sparse edge predictions (consistent with the existing sparsity regularizer).
This is Zygote-safe (no branching on data values).
"""
function edge_supervision_loss(pred_edge, edge_gt; pos_weight::Float32=3.0f0)
    p = clamp.(pred_edge, 1.0f-6, 1.0f0 - 1.0f-6)
    # Weighted BCE: upweight positive (edge) voxels
    edge_bce = -mean(pos_weight .* edge_gt .* log.(p) .+ (1.0f0 .- edge_gt) .* log.(1.0f0 .- p))
    # Dice on edges
    edge_dice = dice_loss(pred_edge, edge_gt)
    return 0.5f0 * edge_bce + 0.5f0 * edge_dice
end
