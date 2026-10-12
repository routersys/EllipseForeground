import numpy as np


def disk_kernel(shape: tuple[int, int], radius: float) -> np.ndarray:
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]
    rows, columns = np.minimum(rows, height - rows), np.minimum(columns, width - columns)
    kernel = ((rows * rows + columns * columns) <= radius * radius).astype(np.float64)
    return kernel / kernel.sum()


def disk_blur(image: np.ndarray, radius: float) -> np.ndarray:
    margin = int(radius * 2) + 2
    padding = ((margin, margin), (margin, margin)) + ((0, 0),) * (image.ndim - 2)
    padded = np.pad(image.astype(np.float64), padding, mode="edge")
    spectrum = np.fft.rfft2(disk_kernel(padded.shape[:2], radius))
    if padded.ndim == 2:
        blurred = np.fft.irfft2(np.fft.rfft2(padded) * spectrum, s=padded.shape[:2])
    else:
        blurred = np.stack(
            [
                np.fft.irfft2(np.fft.rfft2(padded[..., channel]) * spectrum, s=padded.shape[:2])
                for channel in range(padded.shape[2])
            ],
            axis=2,
        )
    return blurred[margin:-margin, margin:-margin]


def blurred_composite(image: np.ndarray, alpha: np.ndarray, radius: float, epsilon: float) -> np.ndarray:
    keep = (1 - alpha)[..., None]
    premultiplied = disk_blur(image * keep, radius)
    coverage = disk_blur(1 - alpha, radius)[..., None]
    return alpha[..., None] * image + keep * premultiplied / np.maximum(coverage, epsilon)
