import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import pandas as pd

import pyloudnorm as pyln


def calculate_r128_metrics(y, sr):
    try:
        y = y.astype("float32")

        meter = pyln.Meter(sr)

        loudness = meter.integrated_loudness(y)
        loudness_range = meter.loudness_range(y)

        # --- Approximate True Peak (oversampling x4) ---
        upsample_factor = 4
        y_upsampled = np.interp(
            np.linspace(0, len(y), len(y) * upsample_factor), np.arange(len(y)), y
        )

        peak_val = np.max(np.abs(y_upsampled))
        true_peak = 20 * np.log10(peak_val + 1e-9)

        return {
            "Integrated Loudness": f"{loudness:.1f} LUFS",
            "Loudness Range": f"{loudness_range:.1f} LU",
            "True Peak": f"{true_peak:.1f} dBTP",
        }

    except Exception:
        return {
            "Integrated Loudness": "Error",
            "Loudness Range": "Error",
            "True Peak": "Error",
        }


def analyze_and_match_vocal(
    ref_file,
    target_file,
    intensity=50,
    onset_sensitivity=0.5,
    smoothing_mode="Balanced",
    apply_tonal=False,
    apply_denoise=False,
    apply_dereverb=False,
    enable_strip_silence=False,
    apply_lowend_cleanup=True,
    apply_declick=False,
):

    # 1. Load Audio Files
    y_ref, sr = librosa.load(ref_file, sr=None)
    y_target, _ = librosa.load(target_file, sr=sr)

    # --- Adaptive LUFS Pre-Gain Alignment ---
    meter = pyln.Meter(sr)

    lufs_ref = meter.integrated_loudness(y_ref.astype("float32"))
    lufs_target = meter.integrated_loudness(y_target.astype("float32"))

    lufs_diff = lufs_ref - lufs_target

    # adaptivní faktor podle rozdílu
    if abs(lufs_diff) > 8:
        factor = 0.85
    elif abs(lufs_diff) > 4:
        factor = 0.70
    else:
        factor = 0.50

    pre_gain_db = lufs_diff * factor
    pre_gain = 10 ** (pre_gain_db / 20.0)

    y_target = y_target * pre_gain

    # --- Adaptive HPF ---
    if apply_lowend_cleanup:
        y_target = apply_adaptive_hpf(y_target, sr)

    # --- Optional Light Denoise ---
    if apply_denoise:
        y_target = apply_light_denoise(y_target, sr)

    # --- Optional Light De-reverb ---
    if apply_dereverb:
        y_target = apply_light_dereverb(y_target, sr)

    # --- Optional Light Declick ---
    if apply_declick:
        y_target = apply_light_declick(y_target, sr)

    # --- Optional Strip Silence ---
    if enable_strip_silence:
        y_target = apply_strip_silence(
            y_target,
            sr,
            silence_threshold_db=-60,
            min_silence_ms=80,
            fade_ms=15,
        )

    # --- Fixed HPF 50 Hz ---
    # y_target = apply_fixed_hpf(y_target, sr, cutoff=50)

    max_len = max(len(y_ref), len(y_target))

    max_len = max(len(y_ref), len(y_target))
    y_ref = librosa.util.fix_length(y_ref, size=max_len)
    y_target = librosa.util.fix_length(y_target, size=max_len)

    hop_length = 256  # Higher resolution for syllable accuracy (approx 5.8ms frames)
    frame_length = 1024

    y_ref_norm = y_ref / (np.max(np.abs(y_ref)) + 1e-6)
    y_target_norm = y_target / (np.max(np.abs(y_target)) + 1e-6)

    # Base RMS tracking
    rms_ref = librosa.feature.rms(
        y=y_ref_norm, frame_length=frame_length, hop_length=hop_length
    )[0]
    rms_target = librosa.feature.rms(
        y=y_target_norm, frame_length=frame_length, hop_length=hop_length
    )[0]

    num_frames = len(rms_ref)

    # 2. ADVANCED SYLLABLE & PHRASE SMOOTHING (Dual-Stage Window)
    # Fast tracking window for catch-up on hot syllables, wide window for natural crossfades
    intensity_factor = intensity / 100.0
    if smoothing_mode == "Smooth":
        fast_sigma = 22.0
        phrase_sigma = 140.0
        final_smooth_sigma = 12
        gate_sigma = 22
        mode_label = "Smooth"
    elif smoothing_mode == "Sharp":
        fast_sigma = 8.0
        phrase_sigma = 55.0
        final_smooth_sigma = 4
        gate_sigma = 10
        mode_label = "Sharp"
    else:
        fast_sigma = 14.0
        phrase_sigma = 95.0
        final_smooth_sigma = 8
        gate_sigma = 16
        mode_label = "Balanced"

    fader_speed = f"Auto {mode_label}"

    # Calculate phrase level trends and transient syllable spikes
    rms_ref_macro = gaussian_filter1d(rms_ref, sigma=phrase_sigma)
    rms_target_macro = gaussian_filter1d(rms_target, sigma=phrase_sigma)

    rms_ref_micro = gaussian_filter1d(rms_ref, sigma=fast_sigma)
    rms_target_micro = gaussian_filter1d(rms_target, sigma=fast_sigma)

    # Spectral flux / onset sensitivity for faster language-independent transient response
    onset_strength_ref = librosa.onset.onset_strength(
        y=y_ref_norm, sr=sr, hop_length=hop_length
    )
    onset_strength_tgt = librosa.onset.onset_strength(
        y=y_target_norm, sr=sr, hop_length=hop_length
    )

    if len(onset_strength_ref) < num_frames:
        onset_strength_ref = np.pad(
            onset_strength_ref,
            (0, num_frames - len(onset_strength_ref)),
            mode="constant",
        )
    else:
        onset_strength_ref = onset_strength_ref[:num_frames]

    if len(onset_strength_tgt) < num_frames:
        onset_strength_tgt = np.pad(
            onset_strength_tgt,
            (0, num_frames - len(onset_strength_tgt)),
            mode="constant",
        )
    else:
        onset_strength_tgt = onset_strength_tgt[:num_frames]

    onset_combined = (onset_strength_ref + onset_strength_tgt) * 0.5
    onset_norm = onset_combined / (np.max(onset_combined) + 1e-6)

    # Use dB-domain ratios for more natural gain matching
    epsilon = 1e-6
    rms_ref_db = 20 * np.log10(rms_ref + epsilon)
    rms_target_db = 20 * np.log10(rms_target + epsilon)
    rms_ref_macro_db = 20 * np.log10(rms_ref_macro + epsilon)
    rms_target_macro_db = 20 * np.log10(rms_target_macro + epsilon)
    rms_ref_micro_db = 20 * np.log10(rms_ref_micro + epsilon)
    rms_target_micro_db = 20 * np.log10(rms_target_micro + epsilon)

    macro_diff_db = rms_ref_macro_db - rms_target_macro_db
    micro_diff_db = rms_ref_micro_db - rms_target_micro_db

    onset_gain = onset_sensitivity
    downward_alpha = np.clip(0.14 + onset_norm * (0.32 + onset_gain * 0.18), 0.14, 0.8)
    upward_alpha = np.clip(
        0.10 + (1.0 - onset_norm) * (0.16 - onset_gain * 0.06), 0.06, 0.28
    )

    smoothed_micro_db = np.zeros_like(micro_diff_db)
    smoothed_micro_db[0] = micro_diff_db[0]
    for i in range(1, num_frames):
        diff = micro_diff_db[i] - smoothed_micro_db[i - 1]
        alpha = downward_alpha[i] if diff < 0 else upward_alpha[i]
        smoothed_micro_db[i] = smoothed_micro_db[i - 1] + alpha * diff

    # Combined target: phrase flow plus micro-syllable correction for transients
    macro_weight = np.clip(0.75 - 0.25 * onset_norm * onset_gain, 0.35, 0.75)
    micro_weight = 1.0 - macro_weight
    pure_gain_db = (macro_weight * macro_diff_db) + (micro_weight * smoothed_micro_db)
    pure_gain_db = np.clip(pure_gain_db, -6.0, 6.0)

    # Scale correction by intensity in dB space, but keep core phrase shape natural
    gain_db = intensity_factor * pure_gain_db
    gain_curve = 10 ** (gain_db / 20.0)

    # Light final smoothing; preserve transient detail while avoiding pumping
    final_sigma = max(2.4, final_smooth_sigma * (1.0 - onset_gain * 0.15))
    gain_curve = gaussian_filter1d(gain_curve, sigma=final_sigma)

    def normalized_onset_strength(signal):
        onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
        if len(onset) < num_frames:
            onset = np.pad(onset, (0, num_frames - len(onset)), mode="constant")
        else:
            onset = onset[:num_frames]
        return onset / (np.max(onset) + 1e-6)

    gain_samples = np.interp(
        np.arange(len(y_target)), np.arange(len(gain_curve)) * hop_length, gain_curve
    )

    # 3. Apply pure volume modification
    y_modulated = y_target * gain_samples

    # Quick self-audit: if output is too aggressive or too muted, adjust gently
    output_onset_norm = normalized_onset_strength(y_modulated)
    onset_ratio = np.mean(output_onset_norm / (onset_norm + 1e-6))
    avg_gain_db = np.mean(np.abs(gain_db))
    positive_gain_avg = np.mean(gain_db[gain_db > 0]) if np.any(gain_db > 0) else 0.0

    if onset_ratio > 1.18 and avg_gain_db > 2.2:
        backoff = 0.82 + 0.08 * (1.0 - np.clip(np.mean(onset_norm), 0.0, 1.0))
        gain_db = gain_db * backoff
    elif onset_ratio < 0.9 and positive_gain_avg > 0.8:
        boost = 1.06 + 0.06 * np.clip((0.9 - onset_ratio) / 0.15, 0.0, 1.0)
        gain_db = np.where(gain_db > 0, gain_db * boost, gain_db)

    if (
        onset_ratio > 1.18
        and avg_gain_db > 2.2
        or onset_ratio < 0.9
        and positive_gain_avg > 0.8
    ):
        gain_curve = 10 ** (gain_db / 20.0)
        gain_curve = gaussian_filter1d(gain_curve, sigma=final_sigma)
        gain_samples = np.interp(
            np.arange(len(y_target)),
            np.arange(len(gain_curve)) * hop_length,
            gain_curve,
        )
        y_modulated = y_target * gain_samples

    # Natural hysteresis gating for silence, breaths and soft tails
    silence_threshold_on = 0.004
    silence_threshold_off = 0.002
    rms_ref_samples = np.interp(
        np.arange(len(y_ref)), np.arange(len(rms_ref_micro)) * hop_length, rms_ref_micro
    )
    rms_target_samples = np.interp(
        np.arange(len(y_target)),
        np.arange(len(rms_target_micro)) * hop_length,
        rms_target_micro,
    )

    gate_state = False
    gate_values = np.zeros(len(y_ref), dtype=float)
    for i in range(len(y_ref)):
        if (
            rms_ref_samples[i] >= silence_threshold_on
            or rms_target_samples[i] >= silence_threshold_on
        ):
            gate_state = True
        elif (
            rms_ref_samples[i] < silence_threshold_off
            and rms_target_samples[i] < silence_threshold_off
        ):
            gate_state = False
        gate_values[i] = 1.0 if gate_state else 0.0

    gate_envelope = gaussian_filter1d(gate_values, sigma=gate_sigma)
    gate_envelope = np.clip(gate_envelope, 0.0, 1.0)
    y_modulated *= gate_envelope

    # Global Energy Trim Match to center the mix perfectly
    meter = pyln.Meter(sr)

    lufs_ref = meter.integrated_loudness(y_ref.astype("float32"))
    lufs_out = meter.integrated_loudness(y_modulated.astype("float32"))

    lufs_diff = lufs_ref - lufs_out

    # jemný stabilizační rebalance (např 70%)
    rebalance_gain_db = lufs_diff * 0.7
    rebalance_gain = 10 ** (rebalance_gain_db / 20.0)

    y_modulated = y_modulated * rebalance_gain

    # --- Frame-based Selective Micro-Peak Leveling (SAFE) ---

    target_peak = np.max(np.abs(y_ref)) * 0.92

    frame_peaks = np.array(
        [
            np.max(np.abs(y_modulated[i : i + frame_length]))
            for i in range(0, len(y_modulated), hop_length)
        ]
    )

    gain_reduction = np.clip(target_peak / (frame_peaks + 1e-9), 0.60, 1.0)

    gain_reduction = gaussian_filter1d(gain_reduction, sigma=3)

    gain_samples = np.interp(
        np.arange(len(y_modulated)),
        np.arange(len(gain_reduction)) * hop_length,
        gain_reduction,
    )

    y_modulated *= gain_samples

    if apply_tonal:
        y_modulated = apply_tonal_matching(y_ref, y_modulated, sr)

    # --- Final Loudness Rebalance (after tonal) ---
    meter = pyln.Meter(sr)

    lufs_ref = meter.integrated_loudness(y_ref.astype("float32"))
    lufs_out = meter.integrated_loudness(y_modulated.astype("float32"))

    lufs_diff = lufs_ref - lufs_out

    rebalance_gain_db = lufs_diff * 1.0
    rebalance_gain = 10 ** (rebalance_gain_db / 20.0)

    y_modulated *= rebalance_gain

    # --- Final Peak Control (after tonal) ---
    target_peak = np.max(np.abs(y_ref)) * 0.92

    frame_peaks = np.array(
        [
            np.max(np.abs(y_modulated[i : i + frame_length]))
            for i in range(0, len(y_modulated), hop_length)
        ]
    )

    gain_reduction = np.clip(target_peak / (frame_peaks + 1e-9), 0.60, 1.0)
    gain_reduction = gaussian_filter1d(gain_reduction, sigma=3)

    gain_samples = np.interp(
        np.arange(len(y_modulated)),
        np.arange(len(gain_reduction)) * hop_length,
        gain_reduction,
    )

    y_modulated *= gain_samples

    times = librosa.times_like(rms_ref_macro, sr=sr, hop_length=hop_length)

    metrics_ref = calculate_r128_metrics(y_ref, sr)
    metrics_target = calculate_r128_metrics(y_target, sr)
    metrics_out = calculate_r128_metrics(y_modulated, sr)

    return (
        y_modulated,
        sr,
        times,
        rms_ref_macro,
        rms_target_macro,
        gain_curve,
        fader_speed,
        intensity,
        metrics_ref,
        metrics_target,
        metrics_out,
    )


def apply_tonal_matching(y_ref, y_target, sr):
    import numpy as np
    import librosa
    from scipy.ndimage import gaussian_filter1d

    # ---- STFT ----
    n_fft = 2048

    ref_stft = np.abs(librosa.stft(y_ref, n_fft=n_fft))
    target_stft = np.abs(librosa.stft(y_target, n_fft=n_fft))

    # ---- average spectrum ----
    ref_avg = np.mean(ref_stft, axis=1)
    target_avg = np.mean(target_stft, axis=1) + 1e-9

    # ---- EQ difference (linear) ----
    eq_curve = ref_avg / target_avg

    # ---- smooth EQ curve ----
    eq_curve = gaussian_filter1d(eq_curve, sigma=1.5)

    # ---- limit extreme EQ ----
    eq_curve = np.clip(eq_curve, 0.4, 2.5)

    # ---- apply with blend ----
    eq_strength = 0.8  # začni konzervativně

    target_stft_complex = librosa.stft(y_target, n_fft=n_fft)
    mag = np.abs(target_stft_complex)
    phase = np.angle(target_stft_complex)

    # blend
    eq_curve = (1 - eq_strength) + eq_strength * eq_curve

    # apply frequency shaping
    mag_matched = mag * eq_curve[:, np.newaxis]

    # reconstruct
    y_out = librosa.istft(mag_matched * np.exp(1j * phase))

    return y_out


# --- WEB INTERFACE ---
st.set_page_config(page_title="AI Vocal Leveler", page_icon="🎤", layout="centered")
st.title("🎤 AI Vocal Leveler")
st.subheader("Automated Volume Dynamics Matching")
st.write(
    "Upload the reference track and your target language track to automatically match the volume dynamics."
)

ref_upload = st.file_uploader(
    "1. Upload Reference Vocal (e.g., English WAV)", type=["wav"]
)
target_upload = st.file_uploader(
    "2. Upload Localized Vocal (e.g., Target Language WAV)", type=["wav"]
)

st.write("---")
st.subheader("🚀 Automatic Processing")

apply_tonal = st.checkbox("🎛 Apply tonal matching (beta)", value=True)

apply_lowend_cleanup = st.checkbox("🎧 Apply low-end cleanup", value=True)

apply_denoise = st.checkbox("🧹 Apply light denoise (beta)", value=True)

apply_dereverb = st.checkbox("🏠 Apply light de-reverb (beta)", value=True)

enable_strip_silence = st.checkbox("✂️ Strip silence (beta)", value=True)

apply_declick = st.checkbox("👄 Light declick (experimental)", value=False)


def apply_adaptive_hpf(y, sr):
    import numpy as np
    import librosa
    from scipy.ndimage import gaussian_filter1d

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
    fundamental_freqs = np.clip(fundamental_freqs, 70, 250)

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

    return y_out


from scipy.signal import butter, filtfilt


def apply_fixed_hpf(y, sr, cutoff=50):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist

    # 4th order → cca ~24 dB/oct → můžeme aplikovat 2x pro větší strmost
    b, a = butter(4, norm_cutoff, btype="highpass")

    y_filtered = filtfilt(b, a, y)

    return y_filtered


def apply_light_denoise(y, sr):
    import numpy as np
    import librosa
    from scipy.ndimage import gaussian_filter1d

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

    return y_out


def apply_light_dereverb(y, sr):
    import numpy as np
    import librosa
    from scipy.ndimage import gaussian_filter1d

    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    mag = np.abs(stft)
    phase = np.angle(stft)

    # dlouhodobá energie = odhad room tailu
    reverb_estimate = gaussian_filter1d(mag, sigma=8, axis=1)

    dereverb_strength = 0.15

    mag_clean = mag - (reverb_estimate * dereverb_strength)
    mag_clean = np.maximum(mag_clean, 0)

    y_out = librosa.istft(mag_clean * np.exp(1j * phase), hop_length=hop_length)

    return y_out


def apply_light_declick(y, sr):
    from scipy.signal import medfilt
    import numpy as np

    # velmi jemný globální declick
    smoothed = medfilt(y, kernel_size=3)

    # pouze lehké přimíchání
    blend = 0.15

    y_out = ((1 - blend) * y) + (blend * smoothed)

    return y_out


def apply_strip_silence(
    y,
    sr,
    silence_threshold_db=-60,
    min_silence_ms=60,
    fade_ms=15,
):
    import numpy as np
    import librosa

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


st.info("💡 Upload files and click process.")

if ref_upload and target_upload:
    if st.button("⚡ Process and Match Volumes", type="primary"):
        with st.spinner("Analyzing syllable structures and generating crossfades..."):
            try:
                (
                    output_audio,
                    sample_rate,
                    times,
                    rms_ref,
                    rms_target,
                    gain_curve,
                    final_speed,
                    final_intensity,
                    m_ref,
                    m_tgt,
                    m_out,
                ) = analyze_and_match_vocal(
                    ref_upload,
                    target_upload,
                    55,
                    0.5,
                    "Balanced",
                    apply_tonal=apply_tonal,
                    apply_denoise=apply_denoise,
                    apply_dereverb=apply_dereverb,
                    enable_strip_silence=enable_strip_silence,
                    apply_lowend_cleanup=apply_lowend_cleanup,
                    apply_declick=apply_declick,
                )

                output_fn = "leveled_target_vocal.wav"
                sf.write(output_fn, output_audio, sample_rate)

                st.success("✓ Audio successfully leveled!")

                st.code(
                    f"AI Song Analysis Completed:\n -> Mode selected: {final_speed}\n -> Applied match intensity: {final_intensity}%"
                )

                # REPOSITIONED AND RENAMED PROFESSIONAL R128 TABLE
                st.subheader("📊 Loudness Analysis (EBU R128 Standard)")
                data_metrics = {
                    "Industry Metric": [
                        "Integrated Loudness",
                        "Loudness Range",
                        "True Peak",
                    ],
                    "1. Reference Vocal (Source)": [
                        m_ref["Integrated Loudness"],
                        m_ref["Loudness Range"],
                        m_ref["True Peak"],
                    ],
                    "2. Localized Vocal (Before Fix)": [
                        m_tgt["Integrated Loudness"],
                        m_tgt["Loudness Range"],
                        m_tgt["True Peak"],
                    ],
                    "3. Output Vocal (After AI Fix)": [
                        m_out["Integrated Loudness"],
                        m_out["Loudness Range"],
                        m_out["True Peak"],
                    ],
                }
                df = pd.DataFrame(data_metrics)
                st.table(df)

                # PLOT GRAPH
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                fig.patch.set_facecolor("#0e1117")

                ax1.set_facecolor("#131722")
                ax1.plot(
                    times,
                    rms_ref,
                    label="Reference Envelope (Destination)",
                    color="#f39c12",
                    linewidth=2,
                )
                ax1.plot(
                    times,
                    rms_target,
                    label="Original Target Envelope",
                    color="#3498db",
                    linewidth=1.5,
                    linestyle="--",
                )
                ax1.set_title(
                    "Volume Envelopes (RMS) Comparison", color="white", fontsize=12
                )
                ax1.legend(loc="upper right")
                ax1.grid(True, color="#2c3e50", linestyle=":")
                ax1.tick_params(colors="white")

                ax2.set_facecolor("#131722")
                ax2.plot(
                    times,
                    gain_curve,
                    label="Applied Gain Automation",
                    color="#2ecc71",
                    linewidth=2,
                )
                ax2.axhline(1.0, color="white", linestyle=":", alpha=0.5)
                ax2.set_title(
                    "Applied Gain Automation Curve (Variable in Time)",
                    color="white",
                    fontsize=12,
                )
                ax2.set_xlabel("Time (seconds)", color="white")
                ax2.set_ylabel("Gain Factor", color="white")
                ax2.legend(loc="upper right")
                ax2.grid(True, color="#2c3e50", linestyle=":")
                ax2.tick_params(colors="white")

                plt.tight_layout()
                st.pyplot(fig)

                # DOWNLOAD SECTION
                st.write("---")
                st.subheader("💾 Download Leveled Output")
                st.caption(
                    "Pre-rendered and fully leveled WAV file ready for the mix. Import this directly back into Pro Tools."
                )

                st.audio(output_fn, format="audio/wav")
                with open(output_fn, "rb") as file:
                    st.download_button(
                        label="🚀 Download Leveled Vocal WAV",
                        data=file,
                        file_name="leveled_target_vocal.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )

                os.remove(output_fn)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
