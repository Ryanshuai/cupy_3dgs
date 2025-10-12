import cupy as cp
from cupyx.scipy.sparse import csr_matrix

from gs_core.render_kernel import render_kernel


def inverse_sigma(sigma_screen):
    det = sigma_screen[:, 0, 0] * sigma_screen[:, 1, 1] - sigma_screen[:, 0, 1] ** 2
    valid = det > 1e-6

    inv_det = 1.0 / cp.where(valid, det, 1.0)
    sigma_inv = cp.stack([
        sigma_screen[:, 1, 1] * inv_det,
        -sigma_screen[:, 0, 1] * inv_det,
        -sigma_screen[:, 0, 1] * inv_det,
        sigma_screen[:, 0, 0] * inv_det
    ], axis=1).reshape(-1, 2, 2)
    return sigma_inv, valid


def get_tile_center(image_w, image_h, tile_size):
    n_tiles_x = (image_w + tile_size - 1) // tile_size
    n_tiles_y = (image_h + tile_size - 1) // tile_size

    tx = cp.arange(n_tiles_x, dtype=cp.int32)
    ty = cp.arange(n_tiles_y, dtype=cp.int32)
    TX, TY = cp.meshgrid(tx, ty)  # (ny, nx)

    tile_x0 = TX * tile_size
    tile_y0 = TY * tile_size
    tile_x1 = cp.minimum(tile_x0 + tile_size, image_w)
    tile_y1 = cp.minimum(tile_y0 + tile_size, image_h)

    tile_centers_x = (tile_x0 + tile_x1) * 0.5
    tile_centers_y = (tile_y0 + tile_y1) * 0.5
    tile_centers = cp.stack([tile_centers_x, tile_centers_y], axis=-1).reshape(-1, 2).astype(cp.float32)
    return tile_centers, n_tiles_x, n_tiles_y


def get_tile_gaussian_indices(tile_centers, tile_size, mu_screen, sigma_screen,
                              tile_chunk=2000, gauss_chunk=100000):
    eigvals = cp.linalg.eigvalsh(sigma_screen)
    gaussian_radius = 3.0 * cp.sqrt(eigvals.max(axis=1))
    threshold = gaussian_radius + tile_size * 0.5

    num_tiles = len(tile_centers)
    num_gauss = len(mu_screen)
    rows, cols = [], []

    for i in range(0, num_tiles, tile_chunk):
        chunk_tiles = tile_centers[i:i + tile_chunk]

        for j in range(0, num_gauss, gauss_chunk):
            mu_chunk = mu_screen[j:j + gauss_chunk, :2]
            thresh_chunk = threshold[j:j + gauss_chunk]

            dxy = chunk_tiles[:, None, :] - mu_chunk[None, :, :]
            in_tile = (cp.abs(dxy) <= thresh_chunk[None, :, None]).all(axis=2)

            tile_idx, gauss_idx = cp.where(in_tile)
            rows.append(tile_idx + i)
            cols.append(gauss_idx + j)

    rows = cp.concatenate(rows)
    cols = cp.concatenate(cols)
    data = cp.ones(len(rows), dtype=cp.int32)

    return csr_matrix((data, (rows, cols)), shape=(num_tiles, num_gauss))


def render(mu_screen, sigma_screen, opacity, color, screen_w, screen_h, tile_size=16):
    sigma_inv, valid = inverse_sigma(sigma_screen)

    mu_screen = mu_screen[valid]
    sigma_inv = sigma_inv[valid]
    color = color[valid]
    opacity = opacity[valid]

    tile_centers, n_tiles_x, n_tiles_y = get_tile_center(screen_w, screen_h, tile_size)
    num_tiles = tile_centers.shape[0]
    in_tile = get_tile_gaussian_indices(tile_centers, tile_size, mu_screen, sigma_screen)
    tile_gaussian_csr = csr_matrix(in_tile.astype(cp.int32))

    output = cp.zeros((num_tiles, tile_size, tile_size, 3), dtype=cp.float32)
    block_size = 256
    grid_size = (num_tiles + block_size - 1) // block_size

    render_kernel((grid_size,), (block_size,), (
        tile_gaussian_csr.indptr.astype(cp.int32),
        tile_gaussian_csr.indices.astype(cp.int32),
        mu_screen.reshape(-1).astype(cp.float32),
        sigma_inv.reshape(-1).astype(cp.float32),
        color.reshape(-1).astype(cp.float32),
        opacity.astype(cp.float32),
        output.reshape(-1),
        num_tiles, tile_size, screen_w, screen_h
    ))

    output = output.reshape(n_tiles_y, n_tiles_x, tile_size, tile_size, 3)
    output = output.transpose(0, 2, 1, 3, 4).reshape(
        n_tiles_y * tile_size, n_tiles_x * tile_size, 3
    )
    output = output[:screen_h, :screen_w]
    return output
