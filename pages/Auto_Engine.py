import joblib
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
import pandas as pd

from engine.dynamic_match import analyze_and_match_vocal

CLICK_MODEL = joblib.load("click_classifier.pkl")


def detect_click_candidates(y, sr, max_clicks=100):
    import numpy as np
    from scipy.signal import find_peaks

    # -------------------------------------------------
    # LOCAL DISCONTINUITY DETECTION
    # -------------------------------------------------

    diff_signal = np.abs(np.diff(y))

    diff_signal = np.concatenate(([0], diff_signal))

    # -------------------------------------------------
    # ROBUST THRESHOLD
    # -------------------------------------------------

    median = np.median(diff_signal)

    mad = np.median(np.abs(diff_signal - median))

    threshold = median + (4 * mad)

    peaks, _ = find_peaks(
        diff_signal,
        height=threshold,
        distance=int(sr * 0.001),
    )

    # -------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------

    candidates = []

    for peak in peaks:

        click_radius = int(sr * 0.006)

        context_radius = int(sr * 0.050)

        start = max(0, peak - click_radius)

        end = min(len(y), peak + click_radius)

        ctx_start = max(0, peak - context_radius)

        ctx_end = min(len(y), peak + context_radius)

        segment = y[start:end]

        context = y[ctx_start:ctx_end]

        if len(segment) < 32:
            continue

        # -------------------------------------------------
        # LOCAL DERIVATIVE SCORE
        # -------------------------------------------------

        seg_diff = np.abs(np.diff(segment))

        anomaly = np.max(seg_diff)

        normal = np.median(seg_diff) + 1e-9

        anomaly_ratio = anomaly / normal

        anomaly_score = np.clip(
            anomaly_ratio / 20.0,
            0,
            1,
        )

        # -------------------------------------------------
        # WIDTH
        # -------------------------------------------------

        active = np.where(seg_diff > (0.25 * anomaly))[0]

        if len(active) > 0:

            width_ms = (len(active) / sr) * 1000

        else:

            width_ms = 100

        if width_ms <= 2:
            width_score = 1.0
        elif width_ms <= 5:
            width_score = 0.9
        elif width_ms <= 8:
            width_score = 0.7
        elif width_ms <= 12:
            width_score = 0.4
        else:
            width_score = 0.1

        # -------------------------------------------------
        # ISOLATION
        # -------------------------------------------------

        click_rms = np.sqrt(np.mean(segment**2))

        context_rms = np.sqrt(np.mean(context**2))

        isolation = click_rms / (context_rms + 1e-9)

        isolation_score = np.clip(
            isolation * 3,
            0,
            1,
        )

        # -------------------------------------------------
        # SHARPNESS
        # -------------------------------------------------

        peak_derivative = np.max(seg_diff)

        mean_derivative = np.mean(seg_diff) + 1e-9

        sharpness = peak_derivative / mean_derivative

        sharpness_score = np.clip(
            sharpness / 15.0,
            0,
            1,
        )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = (
            anomaly_score * 0.45
            + width_score * 0.30
            + sharpness_score * 0.15
            + isolation_score * 0.10
        )

        # -------------------------------------------------
        # PENALTY FOR LONG EVENTS
        # -------------------------------------------------

        if width_ms > 10:
            confidence *= 0.5

        if width_ms > 15:
            confidence *= 0.25

        confidence = float(
            np.clip(
                confidence,
                0,
                1,
            )
        )

        if confidence < 0.25:
            continue

        candidates.append(
            (
                peak / sr,
                confidence,
            )
        )

    # -------------------------------------------------
    # SORT
    # -------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x[1],
        reverse=True,
    )

    # -------------------------------------------------
    # DE-DUP
    # -------------------------------------------------

    results = []

    for t, score in candidates:

        if all(abs(t - old_t) > 0.03 for old_t, _ in results):
            results.append(
                (
                    t,
                    score,
                )
            )

        if len(results) >= max_clicks:
            break

    return results


def create_click_track(
    y,
    sr,
    clicks,
    window_ms=50,
):

    import numpy as np

    out = np.zeros_like(y)

    pad = int(sr * window_ms / 1000)

    for t, score in clicks:

        center = int(t * sr)

        start = max(0, center - pad)
        end = min(len(y), center + pad)

        out[start:end] = y[start:end]

    return out


def extract_click_features(segment, sr):

    import numpy as np
    import librosa

    peak = np.max(np.abs(segment))

    rms = np.sqrt(np.mean(segment**2))

    crest = peak / (rms + 1e-9)

    zcr = np.mean(librosa.feature.zero_crossing_rate(segment))

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=segment,
            sr=sr,
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=segment,
            sr=sr,
        )
    )

    flatness = np.mean(librosa.feature.spectral_flatness(y=segment))

    d1 = np.diff(segment)

    d1_ratio = np.max(np.abs(d1)) / (np.mean(np.abs(d1)) + 1e-9)

    return [
        [
            peak,
            rms,
            crest,
            zcr,
            centroid,
            rolloff,
            flatness,
            d1_ratio,
        ]
    ]


def classify_click_candidates(
    y,
    sr,
    candidates,
):

    results = []

    radius = int(sr * 0.010)

    for t, _ in candidates:

        center = int(t * sr)

        start = max(0, center - radius)

        end = min(len(y), center + radius)

        segment = y[start:end]

        if len(segment) < 32:
            continue

        features = extract_click_features(
            segment,
            sr,
        )

        probability = float(CLICK_MODEL.predict_proba(features)[0][1])

        if probability < 0.80:
            continue

        results.append(
            (
                t,
                probability,
            )
        )

    return results


st.set_page_config(page_title="Vocal Match Engine", page_icon="🎤", layout="centered")
st.title("🎤 Vocal Match Engine")
st.subheader("Reference-Based Vocal Processing")
st.write(
    "Upload a reference vocal and a target vocal to automatically match dynamics, tone, loudness and overall vocal character."
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

apply_declick = st.checkbox("👄 Light declick (experimental)", value=True)

apply_click_detection = st.checkbox(
    "🔍 Click Detection Assistant (beta)",
    value=True,
)

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
                    ref_audio,
                    target_audio,
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

                # -------------------------------------------------
                # CLICK DETECTION ASSISTANT
                # -------------------------------------------------

                candidates = detect_click_candidates(
                    output_audio,
                    sample_rate,
                    max_clicks=1000,
                )

                clicks = classify_click_candidates(
                    output_audio,
                    sample_rate,
                    candidates,
                )

                click_track = create_click_track(
                    output_audio,
                    sample_rate,
                    clicks,
                )

                click_track_fn = "click_track.wav"

                sf.write(
                    click_track_fn,
                    click_track,
                    sample_rate,
                )

                df_clicks = pd.DataFrame(
                    {
                        "Time Seconds": [x[0] for x in clicks],
                        "Confidence": [x[1] for x in clicks],
                    }
                )

                csv_data = df_clicks.to_csv(index=False)

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

                fig, (
                    ax1,
                    ax2,
                ) = plt.subplots(
                    2,
                    1,
                    figsize=(10, 6),
                    sharex=False,
                )

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

                st.write("---")

                st.subheader("🔍 Click Detection Assistant")

                st.download_button(
                    label="📍 Download Click Markers CSV",
                    data=csv_data,
                    file_name="click_markers.csv",
                    mime="text/csv",
                )

                with open(click_track_fn, "rb") as file:
                    st.download_button(
                        label="🎧 Download Click Track WAV",
                        data=file,
                        file_name="click_track.wav",
                        mime="audio/wav",
                    )

                os.remove(output_fn)
                os.remove(click_track_fn)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
