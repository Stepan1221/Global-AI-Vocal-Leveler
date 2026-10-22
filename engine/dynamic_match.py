import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.ndimage import gaussian_filter1d

from engine.tonal_match import apply_tonal_matching
from engine.cleanup import (
    apply_adaptive_hpf,
    apply_subsonic_cleanup,
    apply_light_denoise,
    apply_light_dereverb,
    apply_light_declick,
    apply_strip_silence,
)
from engine.metrics import calculate_r128_metrics


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

        # --- Subsonic Cleanup ---
        y_target = apply_subsonic_cleanup(y_target, sr)

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

    target_peak = np.max(np.abs(y_ref)) * 0.97

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

    # -------------------------------------------------
    # FINAL LRA REBALANCE
    # -------------------------------------------------

    # meter = pyln.Meter(sr)

    # lra_ref = meter.loudness_range(y_ref.astype("float32"))
    # lra_out = meter.loudness_range(y_modulated.astype("float32"))

    # if np.isfinite(lra_ref) and np.isfinite(lra_out) and lra_ref > 0 and lra_out > 0:

    #     lra_ratio = lra_ref / lra_out

    #     correction = np.clip(
    #         lra_ratio,
    #         0.80,
    #         1.20,
    #     )

    #     gain_center = np.mean(gain_curve)

    #     gain_curve_lra = gain_center + ((gain_curve - gain_center) * correction)

    #     gain_curve_lra = np.maximum(
    #         gain_curve_lra,
    #         0.01,
    #     )

    #     gain_samples_lra = np.interp(
    #         np.arange(len(y_modulated)),
    #         np.arange(len(gain_curve_lra)) * hop_length,
    #         gain_curve_lra,
    #     )

    #     y_modulated *= gain_samples_lra

    # -------------------------------------------------
    # FINAL LUFS REBALANCE
    # -------------------------------------------------

    meter = pyln.Meter(sr)

    lufs_ref = meter.integrated_loudness(y_ref.astype("float32"))
    lufs_out = meter.integrated_loudness(y_modulated.astype("float32"))

    lufs_diff = lufs_ref - lufs_out

    rebalance_gain_db = lufs_diff * 1.0
    rebalance_gain = 10 ** (rebalance_gain_db / 20.0)

    y_modulated *= rebalance_gain

    # -------------------------------------------------
    # FINAL PEAK CONTROL
    # -------------------------------------------------

    target_peak = np.max(np.abs(y_ref)) * 0.99

    frame_peaks = np.array(
        [
            np.max(np.abs(y_modulated[i : i + frame_length]))
            for i in range(0, len(y_modulated), hop_length)
        ]
    )

    gain_reduction = np.clip(
        target_peak / (frame_peaks + 1e-9),
        0.60,
        1.0,
    )

    gain_reduction = gaussian_filter1d(
        gain_reduction,
        sigma=3,
    )

    gain_samples = np.interp(
        np.arange(len(y_modulated)),
        np.arange(len(gain_reduction)) * hop_length,
        gain_reduction,
    )

    y_modulated *= gain_samples

    times = librosa.times_like(rms_ref_macro, sr=sr, hop_length=hop_length)

    meter = pyln.Meter(sr)

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
        y_ref,
        y_target,
    )
