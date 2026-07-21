import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d


def apply_adaptive_hpf(y, sr):
    n_fft = 2048
    hop_length = 512

    # --- STFT ---
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # --- low band pro detekci ---
    low_band = freqs < 300

    n_frames = mag.shape[1]
    fundamental_freqs = np.zeros(n_frames)

    # --- frame-based fundamental detection ---
    for t in range(n_frames):
        spectrum = mag[:, t]
        idx = np.argmax(spectrum * low_band)
        fundamental_freqs[t] = freqs[idx]

    # --- stabilizace ---
    fundamental_freqs = np.clip(fundamental_freqs, 50, 250)

    # --- smoothing ---
    fundamental_freqs = gaussian_filter1d(fundamental_freqs, sigma=10)

    # --- cutoff ---
    safety_margin = 20
    cutoff = np.maximum(40, fundamental_freqs * 0.6 - safety_margin)
    cutoff = np.minimum(cutoff, 200)

    # --- soft mask ---
    mask = np.zeros_like(mag)
    transition_width = 20  # měkký přechod

    for t in range(n_frames):
        mask[:, t] = np.clip((freqs - cutoff[t]) / transition_width, 0, 1)

    # --- aplikace ---
    mag_filtered = mag * mask

    # --- rekonstukce ---
    y_out = librosa.istft(mag_filtered * np.exp(1j * phase), hop_length=hop_length)

    # Memory cleanup: delete large spectral arrays
    del stft, mag, phase, mask, mag_filtered

    return y_out


def apply_subsonic_cleanup(y, sr):
    n_fft = 4096
    hop_length = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    mag = np.abs(stft)
    phase = np.angle(stft)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mask = np.ones_like(freqs)

    # úplné ticho pod 30 Hz
    mask[freqs < 30] = 0

    # plynulý přechod 30-50 Hz
    transition = (freqs >= 30) & (freqs <= 50)

    mask[transition] = (freqs[transition] - 30) / 20

    mag *= mask[:, np.newaxis]

    y_out = librosa.istft(mag * np.exp(1j * phase), hop_length=hop_length)

    # Memory cleanup: delete large spectral arrays
    del stft, mag, phase, mask

    return y_out


def apply_light_denoise(y, sr):
    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    mag = np.abs(stft)
    phase = np.angle(stft)

    # --- noise profile ---
    frame_energy = np.mean(mag, axis=0)
    noise_frames = frame_energy <= np.percentile(frame_energy, 10)

    if np.any(noise_frames):
        noise_profile = np.median(mag[:, noise_frames], axis=1)
    else:
        noise_profile = np.zeros(mag.shape[0])

    # --- frequency weighting ---
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    weights = np.ones_like(freqs)

    weights[freqs < 200] = 0.10
    weights[(freqs >= 200) & (freqs < 1000)] = 0.40
    weights[(freqs >= 1000) & (freqs < 4000)] = 0.85
    weights[freqs >= 4000] = 1.00

    # --- adaptive strength by loudness ---
    frame_strength = 1 - (frame_energy / (np.max(frame_energy) + 1e-9))

    frame_strength = np.clip(frame_strength, 0.1, 1.0)

    denoise_strength = 0.25

    reduction = (
        noise_profile[:, np.newaxis]
        * weights[:, np.newaxis]
        * frame_strength[np.newaxis, :]
        * denoise_strength
    )

    mag_clean = mag - reduction
    mag_clean = np.maximum(mag_clean, 0)

    # --- smoothing ---
    mag_clean = gaussian_filter1d(mag_clean, sigma=1, axis=1)

    y_out = librosa.istft(mag_clean * np.exp(1j * phase), hop_length=hop_length)

    # Memory cleanup: delete large spectral arrays and noise profile
    del stft, mag, phase, noise_profile, reduction, mag_clean

    return y_out


def apply_light_dereverb(y, sr):
    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    mag = np.abs(stft)
    phase = np.angle(stft)

    # dlouhodobá energie = odhad room tailu
    reverb_estimate = gaussian_filter1d(mag, sigma=8, axis=1)

    dereverb_strength = 0.12

    mag_clean = mag - (reverb_estimate * dereverb_strength)
    mag_clean = np.maximum(mag_clean, 0)

    y_out = librosa.istft(mag_clean * np.exp(1j * phase), hop_length=hop_length)

    # Memory cleanup: delete large spectral arrays and reverb estimate
    del stft, mag, phase, reverb_estimate, mag_clean

    return y_out


def apply_light_declick(y, sr):
    from scipy.signal import medfilt

    # velmi jemný globální declick
    smoothed = medfilt(y, kernel_size=5)

    # lehké přimíchání
    blend = 0.25

    y_out = ((1 - blend) * y) + (blend * smoothed)

    return y_out


def apply_strip_silence(
    y,
    sr,
    silence_threshold_db=-60,
    min_silence_ms=60,
    fade_ms=15,
):
    # --- RMS analýza po 20 ms oknech ---
    frame_ms = 20

    frame_length = int(sr * frame_ms / 1000)
    hop_length = frame_length

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    threshold = 10 ** (silence_threshold_db / 20)

    silent_frames = rms < threshold

    min_frames = max(1, int(min_silence_ms / frame_ms))

    fade_samples = int(sr * fade_ms / 1000)

    y_out = y.copy()

    stripped_regions = 0

    start = None

    for i, is_silent in enumerate(silent_frames):
        if is_silent and start is None:
            start = i
        elif not is_silent and start is not None:
            length = i - start

            if length >= min_frames:
                stripped_regions += 1

                start_sample = start * hop_length
                end_sample = i * hop_length

                fade_len = min(fade_samples, max(1, (end_sample - start_sample) // 2))

                # celý region na nulu
                y_out[start_sample:end_sample] = 0

                # fade out
                fade_out = np.linspace(1, 0, fade_len)

                y_out[start_sample : start_sample + fade_len] = (
                    y[start_sample : start_sample + fade_len] * fade_out
                )

                # fade in
                fade_in = np.linspace(0, 1, fade_len)

                y_out[end_sample - fade_len : end_sample] = (
                    y[end_sample - fade_len : end_sample] * fade_in
                )

            start = None

    print(f"Strip Silence removed {stripped_regions} regions")

    return y_out
