import numpy as np

from src.features.histograms import HistogramFeatureExtractor


def test_feature_dim_matches_advertised():
    fx = HistogramFeatureExtractor(intensity_bins=32, gradient_bins=32, scales=(1.0, 0.5))
    img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    out = fx(img)
    assert out.shape == (fx.dim,)
    assert out.dtype == np.float32


def test_each_subhist_normalised():
    fx = HistogramFeatureExtractor(intensity_bins=16, gradient_bins=16, scales=(1.0,))
    img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    out = fx(img)
    intensity = out[:16]
    gradient = out[16:]
    assert abs(intensity.sum() - 1.0) < 1e-5
    assert abs(gradient.sum() - 1.0) < 1e-5
