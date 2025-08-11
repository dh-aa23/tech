import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
from scipy.ndimage import gaussian_filter1d




def gaussian_smoothing(data, sigma=2):
    """Gaussian Filter Smoothing without look-ahead bias using causal smoothing"""
    smoothed_data = np.full(len(data), np.nan)
    for i in range(sigma, len(data)):
        smoothed_data[i] = gaussian_filter1d(data[:i + 1], sigma=sigma)[-1]
    return smoothed_data




def savitzky_golay_filter(data, window=5, polyorder=2):
    """Savitzky-Golay Filter without look-ahead bias using causal smoothing"""
    smoothed_data = np.full(len(data), np.nan)
    half_window = (window - 1) // 2
    for i in range(half_window, len(data)):
        smoothed_data[i] = savgol_filter(data[i - window + 1:i + 1], window_length=window, polyorder=polyorder)[-1]
    return smoothed_data



def kalman_filter_denoise(data):
    """Kalman Filter Smoothing without look-ahead bias"""
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(data, n_iter=5)
    smoothed_data = np.full(len(data), np.nan)
    for i in range(1, len(data)):
        state_means, _ = kf.filter(data[:i + 1])
        smoothed_data[i] = state_means[-1, 0]
    return smoothed_data




def fourier_lowpass(data, cutoff=0.1, window=256):
    """Fourier Transform Low-Pass Filter using a rolling window to avoid look-ahead bias"""
    denoised_data = np.full(len(data), np.nan)
    for i in range(window, len(data)):
        window_data = data[i - window:i]
        fft = np.fft.fft(window_data)
        frequencies = np.fft.fftfreq(len(window_data))
        fft[np.abs(frequencies) > cutoff] = 0
        denoised_data[i] = np.fft.ifft(fft).real[-1]
    return denoised_data

