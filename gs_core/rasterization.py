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


def get_tile_gaussian_indices(tile_centers, tile_size, mu_screen, sigma_screen):
    eigvals = cp.linalg.eigvalsh(sigma_screen)  # (N, 2)
    gaussian_radius = 3.0 * cp.sqrt(cp.max(eigvals, axis=1))  # (N,)  max for consider rotation
    gaussian_wh = cp.repeat(gaussian_radius[:, None], 2, axis=1)  # (N, 2)
    gaussian_wh_marge = gaussian_wh + tile_size * 0.5  # (N, 2)
    gaussian_wh_marge = gaussian_wh_marge[None, :, :]  # (1, N, 2)

    tile_centers = tile_centers[:, None, :]  # (num_tiles, 1, 2)
    mu_screen = mu_screen[None, :, :2]  # (1, N, 2)
    dxy = tile_centers - mu_screen  # (num_tiles, N, 2)

    in_tile = (cp.abs(dxy) <= gaussian_wh_marge).all(axis=2)  # (num_tiles, N)
    return in_tile


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