import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d


def apply_tonal_matching(y_ref, y_target, sr):
    # ---- STFT (compute complex once, extract magnitude/phase) ----
    n_fft = 2048

    # Compute complex STFT once per signal to avoid redundant FFT
    ref_stft_complex = librosa.stft(y_ref, n_fft=n_fft)
    target_stft_complex = librosa.stft(y_target, n_fft=n_fft)

    # Extract magnitudes for EQ curve calculation
    ref_mag = np.abs(ref_stft_complex)
    target_mag = np.abs(target_stft_complex)

    # ---- average spectrum ----
    ref_avg = np.mean(ref_mag, axis=1)
    target_avg = np.mean(target_mag, axis=1) + 1e-9

    # ---- EQ difference (linear) ----
    eq_curve = ref_avg / target_avg

    # ---- smooth EQ curve ----
    eq_curve = gaussian_filter1d(eq_curve, sigma=2.0)

    # ---- limit extreme EQ ----
    eq_curve = np.clip(eq_curve, 0.5, 2.0)

    # ---- apply with blend ----
    eq_strength = 0.7

    # Reuse already-computed complex STFT (no redundant computation)
    mag = np.abs(target_stft_complex)
    phase = np.angle(target_stft_complex)

    # ---- blend ----
    eq_curve = (1 - eq_strength) + eq_strength * eq_curve

    # ---- apply frequency shaping ----
    mag_matched = mag * eq_curve[:, np.newaxis]

    # ---- reconstruct ----
    y_out = librosa.istft(mag_matched * np.exp(1j * phase))

    # Memory cleanup: delete large temporary arrays
    del (
        ref_stft_complex,
        target_stft_complex,
        ref_mag,
        target_mag,
        mag,
        phase,
        eq_curve,
        mag_matched,
    )

    return y_out
