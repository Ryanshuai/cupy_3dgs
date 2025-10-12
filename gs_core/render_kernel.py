import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix


render_kernel = cp.RawKernel(r'''
extern "C" __global__
void render_tiles(
    const int* indptr,
    const int* indices,
    const float* mu,
    const float* sigma_inv,   // packed as [s00, s01, s10, s11] per gaussian
    const float* colors,      // packed as RGB per gaussian
    const float* opacities,   // per gaussian
    float* output,            // per-tile framebuffer, RGB per pixel
    int num_tiles,
    int tile_size,
    int img_width,
    int img_height
) {
    const float kAlphaEps = 1.0f / 255.0f;  // ~0.0039215686
    const float kAlphaCap = 0.99f;          // clamp alpha to avoid full opacity
    const float kStopT    = 1e-4f;          // early-exit threshold on transmittance

    int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= num_tiles) return;

    int tiles_per_row = (img_width + tile_size - 1) / tile_size;
    int tile_y = tile_id / tiles_per_row;
    int tile_x = tile_id % tiles_per_row;
    int tile_start_x = tile_x * tile_size;
    int tile_start_y = tile_y * tile_size;

    const int start = indptr[tile_id];
    const int end   = indptr[tile_id + 1];

    for (int py = 0; py < tile_size; ++py) {
        for (int px = 0; px < tile_size; ++px) {
            const int img_x = tile_start_x + px;
            const int img_y = tile_start_y + py;
            if (img_x >= img_width || img_y >= img_height) continue;

            float pixel_color_r = 0.0f;
            float pixel_color_g = 0.0f;
            float pixel_color_b = 0.0f;
            float T = 1.0f;

            // pixel center
            const float px_c = (float)img_x + 0.5f;
            const float py_c = (float)img_y + 0.5f;

            for (int i = start; i < end; ++i) {
                const int gid = indices[i];

                const float dx = px_c - mu[gid * 2 + 0];
                const float dy = py_c - mu[gid * 2 + 1];

                const float inv_s00 = sigma_inv[gid * 4 + 0];
                const float inv_s01 = sigma_inv[gid * 4 + 1];
                const float inv_s11 = sigma_inv[gid * 4 + 3];

                // Gaussian exponent
                const float power = -0.5f * (dx * dx * inv_s00 +
                                             2.0f * dx * dy * inv_s01 +
                                             dy * dy * inv_s11);

                // Skip if exponent positive (negligible contribution)
                if (power > 0.0f) continue;

                // Alpha with cap (matches original behavior)
                float alpha = opacities[gid] * __expf(power);
                alpha = fminf(kAlphaCap, alpha);

                // Alpha threshold
                if (alpha < kAlphaEps) continue;

                // Accumulate using current T
                const float weight = alpha * T;
                pixel_color_r += colors[gid * 3 + 0] * weight;
                pixel_color_g += colors[gid * 3 + 1] * weight;
                pixel_color_b += colors[gid * 3 + 2] * weight;

                // Compute tentative next T and early-stop BEFORE updating T
                const float test_T = T * (1.0f - alpha);
                if (test_T < kStopT) {
                    // Do not update T; terminate contributions for this pixel
                    break;
                }
                T = test_T;
            }

            const int out_idx = (tile_id * tile_size * tile_size +
                                 py * tile_size + px) * 3;
            output[out_idx + 0] = pixel_color_r;
            output[out_idx + 1] = pixel_color_g;
            output[out_idx + 2] = pixel_color_b;
        }
    }
}

''', 'render_tiles')

if __name__ == "__main__":

    # ==== Test setup ====
    cp.random.seed(0)
    np.random.seed(0)

    # Image dimensions and tiling parameters
    W, H = 96, 64
    tile = 8
    tiles_x = (W + tile - 1) // tile
    tiles_y = (H + tile - 1) // tile
    num_tiles = tiles_x * tiles_y

    # Number of Gaussians
    G = 32
    thr = 0.004  # same threshold as in kernel

    # ==== Random Gaussian setup ====
    mu = cp.stack([
        cp.random.uniform(0.0, W, size=G),
        cp.random.uniform(0.0, H, size=G)
    ], axis=1).astype(cp.float32)  # Gaussian centers (G,2)

    # Build random positive-definite covariance matrices
    A = cp.random.normal(0, 1.0, size=(G, 2, 2)).astype(cp.float32)
    Sigma = A @ cp.transpose(A, (0, 2, 1)) + cp.eye(2, dtype=cp.float32)[None, :, :] * 0.5
    Sigma *= 2.0
    Sigma_inv = cp.linalg.inv(Sigma).astype(cp.float32)

    # Flatten sigma_inv as expected by kernel (row-major)
    sigma_inv_flat = cp.stack([
        Sigma_inv[:, 0, 0],
        Sigma_inv[:, 0, 1],
        Sigma_inv[:, 1, 0],
        Sigma_inv[:, 1, 1]
    ], axis=1).reshape(-1).astype(cp.float32)

    # Random color and opacity
    colors = cp.clip(cp.random.uniform(0, 1, (G, 3)).astype(cp.float32), 0, 1)
    opacities = cp.random.uniform(0.1, 0.9, (G,)).astype(cp.float32)

    # ==== Build sparse tile→gaussian mapping (CSR) ====
    # Use 3σ radius approximation to find overlapping tiles
    evals = cp.linalg.eigvalsh(Sigma)
    r = 3.0 * cp.sqrt(evals[:, 1]).astype(cp.float32)

    tile_gid_lists = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0, y0 = tx * tile, ty * tile
            x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
            cx, cy = mu[:, 0], mu[:, 1]

            # Compute nearest distance between Gaussian center and tile box
            nx = cp.maximum(cp.maximum(x0 - cx, 0), cx - x1)
            ny = cp.maximum(cp.maximum(y0 - cy, 0), cy - y1)
            dist = cp.sqrt(nx * nx + ny * ny)
            keep = (dist <= r)
            gids = cp.where(keep)[0].astype(cp.int32)
            tile_gid_lists.append(gids)

    # Sort indices (optional: simulates depth ordering)
    tile_gid_lists = [gids[cp.argsort(gids)] for gids in tile_gid_lists]

    # Build CSR arrays
    indptr = [0]
    indices = []
    for gids in tile_gid_lists:
        indices.append(gids)
        indptr.append(indptr[-1] + int(gids.size))
    indices = cp.concatenate(indices) if len(indices) > 0 else cp.array([], dtype=cp.int32)
    indptr = cp.asarray(indptr, dtype=cp.int32)

    # ==== Run CUDA kernel ====
    out_tile_buf = cp.zeros((num_tiles * tile * tile * 3,), dtype=cp.float32)
    threads = 128
    blocks = (num_tiles + threads - 1) // threads

    render_kernel((blocks,), (threads,), (
        indptr, indices,
        mu.reshape(-1), sigma_inv_flat,
        colors.reshape(-1), opacities,
        out_tile_buf,
        num_tiles, tile, W, H
    ))

    # ==== Reconstruct full image from tile outputs ====
    img_kernel = cp.zeros((H, W, 3), dtype=cp.float32)
    for tile_id in range(num_tiles):
        ty = tile_id // tiles_x
        tx = tile_id % tiles_x
        x0, y0 = tx * tile, ty * tile
        x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
        w, h = x1 - x0, y1 - y0
        base = tile_id * tile * tile * 3
        block = out_tile_buf[base: base + tile * tile * 3].reshape(tile, tile, 3)
        img_kernel[y0:y1, x0:x1, :] = block[:h, :w, :]

    # ==== Reference implementation (vectorized, CPU-equivalent logic) ====
    yy, xx = cp.meshgrid(
        cp.arange(H, dtype=cp.float32) + 0.5,
        cp.arange(W, dtype=cp.float32) + 0.5,
        indexing='ij'
    )

    img_ref = cp.zeros((H, W, 3), dtype=cp.float32)
    T_prev = cp.ones((H, W), dtype=cp.float32)

    for g in range(G):
        dx = xx - mu[g, 0]
        dy = yy - mu[g, 1]
        inv = Sigma_inv[g]
        power = -0.5 * (dx * dx * inv[0, 0] + 2 * dx * dy * inv[0, 1] + dy * dy * inv[1, 1])
        mask = (power <= 0.0)
        alpha = opacities[g] * cp.exp(cp.where(mask, power, cp.zeros_like(power)))
        alpha = cp.where(alpha >= thr, alpha, cp.zeros_like(alpha))
        weight = alpha * T_prev
        img_ref += weight[..., None] * colors[g]
        T_prev = T_prev * (1.0 - alpha)

    # ==== Error metrics ====
    abs_err = cp.abs(img_ref - img_kernel)
    max_err = float(abs_err.max().get())
    mean_err = float(abs_err.mean().get())

    print(f"max |ref - kernel| = {max_err:.6e}")
    print(f"mean |ref - kernel| = {mean_err:.6e}")

    # Basic validation
    assert max_err < 5e-3, "Max error too large!"
    assert not cp.isnan(img_kernel).any(), "Kernel output has NaNs"
    assert img_kernel.min() >= -1e-3 and img_kernel.max() <= 1.001, "Output out of expected [0,1] range"

    print("Test passed ✅")
