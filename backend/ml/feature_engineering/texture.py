import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy


def _entropy(values: np.ndarray) -> float:
    histogram, _ = np.histogram(values, bins=32, range=(0, 255), density=True)
    histogram = histogram[histogram > 0]
    return float(-(histogram * np.log2(histogram)).sum())


def _ridge_density(image: np.ndarray) -> float:
    thresholded = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )
    return float(np.count_nonzero(thresholded) / thresholded.size)


def extract_texture_features(enhanced: np.ndarray) -> dict[str, float]:
    lbp = local_binary_pattern(enhanced, P=16, R=2, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18), density=True)

    quantized = (enhanced / 32).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2],
        levels=8,
        symmetric=True,
        normed=True,
    )

    sobel_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    return {
        "lbp_uniformity": float(np.sum(lbp_hist**2)),
        "lbp_peak": float(np.max(lbp_hist)),
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "ridge_density": _ridge_density(enhanced),
        "ridge_strength": float(magnitude.mean() / 255.0),
        "intensity_entropy": _entropy(enhanced),
        "intensity_mean": float(enhanced.mean() / 255.0),
        "intensity_std": float(enhanced.std() / 255.0),
    }


def extract_fusion_texture_vector(enhanced: np.ndarray) -> np.ndarray:
    enhanced = cv2.resize(enhanced, (224, 224))

    lbp = local_binary_pattern(enhanced, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10),
        density=True,
    )
    lbp_hist = np.nan_to_num(lbp_hist)

    glcm = graycomatrix(
        enhanced,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_features = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        values = graycoprops(glcm, prop)[0]
        glcm_features.append(np.mean(values))
        glcm_features.append(np.std(values))

    edges = cv2.Canny(enhanced, 50, 150)
    ridge_density = np.mean(edges > 0)

    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    orientation = np.arctan2(gy, gx)

    ridge_features = np.array(
        [
            ridge_density,
            np.mean(magnitude) / 255.0,
            np.mean(enhanced) / 255.0,
            np.std(enhanced) / 255.0,
            shannon_entropy(enhanced) / 8.0,
            np.mean(cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0),
            np.abs(np.mean(np.exp(1j * orientation))),
            cv2.Laplacian(enhanced, cv2.CV_64F).var() / 10000.0,
        ],
        dtype=np.float32,
    )

    vector = np.concatenate(
        [
            lbp_hist,
            np.array(glcm_features, dtype=np.float32),
            ridge_features,
        ]
    )

    return np.nan_to_num(vector).astype(np.float32)


def extract_fusion_texture_vector_multilbp(enhanced: np.ndarray) -> np.ndarray:
    enhanced = cv2.resize(enhanced, (300, 300))

    lbp_features = []
    for p_value, radius in [(8, 1), (16, 2), (24, 3)]:
        lbp = local_binary_pattern(enhanced, P=p_value, R=radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, p_value + 3),
            range=(0, p_value + 2),
            density=True,
        )
        lbp_features.append(np.nan_to_num(hist))

    glcm = graycomatrix(
        enhanced,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    glcm_features = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        values = graycoprops(glcm, prop)[0]
        glcm_features.append(np.mean(values))
        glcm_features.append(np.std(values))

    edges = cv2.Canny(enhanced, 50, 150)
    ridge_density = np.mean(edges > 0)

    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    ridge_features = np.array(
        [
            ridge_density,
            np.mean(magnitude) / 255.0,
            np.mean(enhanced) / 255.0,
            np.std(enhanced) / 255.0,
            shannon_entropy(enhanced) / 8.0,
            np.mean(cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0),
            np.abs(np.mean(np.exp(1j * orientation))),
            cv2.Laplacian(enhanced, cv2.CV_64F).var() / 10000.0,
        ],
        dtype=np.float32,
    )

    vector = np.concatenate(
        [
            np.concatenate(lbp_features),
            np.array(glcm_features, dtype=np.float32),
            ridge_features,
        ]
    )

    return np.nan_to_num(vector).astype(np.float32)
