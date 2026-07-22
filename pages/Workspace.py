import io

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from core.audio_state import (
    init_audio_state,
    get_working_audio,
    get_reference_audio,
    get_source_audio,
    set_original_audio,
    set_working_audio,
    set_reference_audio,
    set_sample_rate,
    has_working_audio,
    has_reference_audio,
    set_original_audio,
    get_original_audio,
    has_original_audio,
)

from engine.cleanup import (
    apply_adaptive_hpf,
    apply_subsonic_cleanup,
    apply_light_denoise,
    apply_light_dereverb,
    apply_light_declick,
    apply_strip_silence,
)
from engine.dynamic_match import analyze_and_match_vocal
from engine.metrics import calculate_r128_metrics
from engine.tonal_match import apply_tonal_matching

init_audio_state()

if "preview_audio" not in st.session_state:
    st.session_state["preview_audio"] = None

if "preview_sample_rate" not in st.session_state:
    st.session_state["preview_sample_rate"] = None

if "tonal_preview_audio" not in st.session_state:
    st.session_state["tonal_preview_audio"] = None

if "tonal_preview_sample_rate" not in st.session_state:
    st.session_state["tonal_preview_sample_rate"] = None

if "cleanup_preview_audio" not in st.session_state:
    st.session_state["cleanup_preview_audio"] = None

if "cleanup_preview_sample_rate" not in st.session_state:
    st.session_state["cleanup_preview_sample_rate"] = None


def _audio_to_wav_bytes(audio, sample_rate):
    """Convert an audio array into a WAV buffer for the DSP engine and preview UI."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer


def _audio_to_wav_bytes_data(audio, sample_rate):
    """Return WAV audio as raw bytes for playback and file download."""
    preview_buffer = _audio_to_wav_bytes(audio, sample_rate)
    return preview_buffer.getvalue()


def _render_preview_section(audio, sample_rate, title, download_key, file_name):
    """Render a reusable preview player and download control for workspace modules."""
    if audio is None:
        return

    st.subheader(title)

    preview_wav_bytes = _audio_to_wav_bytes_data(audio, sample_rate)
    st.audio(preview_wav_bytes, format="audio/wav")

    st.download_button(
        label="⬇ Download Preview",
        data=preview_wav_bytes,
        file_name=file_name,
        mime="audio/wav",
        key=download_key,
    )


st.title("🛠 Workspace")

st.write("Interactive processing workspace.")

# ----------------------------------
# Uploads
# ----------------------------------

working_upload = st.file_uploader(
    "Working Audio",
    type=["wav"],
    key="workspace_working",
)

reference_upload = st.file_uploader(
    "Reference Audio (optional)",
    type=["wav"],
    key="workspace_reference",
)

# ----------------------------------
# Load Working Audio
# ----------------------------------

if working_upload:

    y, sr = librosa.load(
        working_upload,
        sr=None,
        mono=True,
    )

    set_original_audio(y)
    set_working_audio(y)
    set_sample_rate(sr)

# ----------------------------------
# Load Reference Audio
# ----------------------------------

if reference_upload:

    y_ref, _ = librosa.load(
        reference_upload,
        sr=None,
        mono=True,
    )

    set_reference_audio(y_ref)

# ----------------------------------
# Status
# ----------------------------------

st.subheader("Current Session")

if has_working_audio():
    st.success("Working Audio: Loaded")
else:
    st.error("Working Audio: Not Loaded")

if has_reference_audio():
    st.success("Reference Audio: Loaded")
else:
    st.warning("Reference Audio: Not Loaded")

from core.audio_state import get_sample_rate

sr = get_sample_rate()

if sr and has_working_audio():

    st.subheader("Audio Info")

    st.write(f"Sample Rate: {sr:,} Hz")

    current_audio = get_working_audio()

    st.write(f"Working Audio Samples: {len(current_audio):,}")

    from core.audio_state import has_original_audio

if has_original_audio():
    st.success("Original Audio: Loaded")
else:
    st.error("Original Audio: Not Loaded")

st.divider()

st.header("🎚 Dynamic Match")

source_mode = st.radio(
    "Source Audio",
    [
        "Current Working Audio",
        "Original Audio",
    ],
    key="dynamic_source",
)

run_dynamic = st.button(
    "Run Dynamic Match",
    key="run_dynamic_match",
)

accept_dynamic_preview = st.button(
    "Accept",
    key="accept_dynamic_preview",
)

if run_dynamic:
    source_audio = get_source_audio(source_mode)
    reference_audio = get_reference_audio()
    current_sr = get_sample_rate()

    if source_audio is None:
        st.error("Please load a source audio before running Dynamic Match.")
    elif reference_audio is None:
        st.error("Please load a reference audio before running Dynamic Match.")
    elif current_sr is None:
        st.error("Please ensure the working audio has been loaded before processing.")
    else:
        with st.spinner("Running Dynamic Match..."):
            try:
                # Convert the selected source and reference audio into WAV buffers so
                # the DSP engine can process them via its existing file-based API.
                source_buffer = _audio_to_wav_bytes(source_audio, current_sr)
                reference_buffer = _audio_to_wav_bytes(reference_audio, current_sr)

                (
                    output_audio,
                    output_sr,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = analyze_and_match_vocal(
                    reference_buffer,
                    source_buffer,
                    55,
                    0.5,
                    "Balanced",
                )

                # Store the preview separately so the main working audio is only
                # replaced after the user explicitly accepts it.
                st.session_state["preview_audio"] = output_audio
                st.session_state["preview_sample_rate"] = output_sr

                st.success("Preview Ready")
                st.caption(
                    "The preview was generated without updating the working audio."
                )
            except Exception as exc:
                st.error(f"Dynamic Match failed: {exc}")

# ----------------------------------
# Dynamic Match Compare Section
# ----------------------------------

st.subheader("🎧 Compare")

compare_dynamic_mode = st.radio(
    "Listen To",
    [
        "Original Audio",
        "Dynamic Match Preview",
        "Current Working Audio",
    ],
    key="dynamic_compare_mode",
    horizontal=True,
)

# Select the single audio source for the compare player so the UI remains simple.
if compare_dynamic_mode == "Original Audio":
    compare_audio = get_original_audio()
    compare_sr = get_sample_rate()
elif compare_dynamic_mode == "Dynamic Match Preview":
    compare_audio = st.session_state.get("preview_audio")
    compare_sr = st.session_state.get("preview_sample_rate") or get_sample_rate()
else:
    compare_audio = get_working_audio()
    compare_sr = get_sample_rate()

if compare_audio is not None:
    st.audio(_audio_to_wav_bytes_data(compare_audio, compare_sr), format="audio/wav")
else:
    st.info("No audio is available for the selected compare option yet.")

if st.session_state.get("preview_audio") is not None:
    preview_wav_bytes = _audio_to_wav_bytes_data(
        st.session_state["preview_audio"],
        st.session_state.get("preview_sample_rate") or get_sample_rate(),
    )

    st.download_button(
        label="⬇ Download Preview",
        data=preview_wav_bytes,
        file_name="dynamic_preview.wav",
        mime="audio/wav",
        key="download_dynamic_preview_wav",
    )

    if accept_dynamic_preview:
        set_working_audio(st.session_state["preview_audio"])
        st.success("Working audio updated from the dynamic preview.")
# ----------------------------------
# Dynamic Match Analysis Panel
# ----------------------------------

with st.expander("▶ Show Dynamics Analysis", expanded=False):
    reference_audio = get_reference_audio()
    dynamic_preview_audio = st.session_state.get("preview_audio")
    current_sr = get_sample_rate()

    if reference_audio is not None and dynamic_preview_audio is not None and current_sr:
        ref_metrics = calculate_r128_metrics(reference_audio, current_sr)
        preview_metrics = calculate_r128_metrics(dynamic_preview_audio, current_sr)

        st.write("### Loudness Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Reference Audio**")
            for metric_name, metric_value in ref_metrics.items():
                st.write(f"- {metric_name}: {metric_value}")

        with col2:
            st.markdown("**Dynamic Match Preview**")
            for metric_name, metric_value in preview_metrics.items():
                st.write(f"- {metric_name}: {metric_value}")
    else:
        st.info("Generate a Dynamic Match preview to view analysis metrics.")

st.divider()

st.header("🎛 Tonal Match")

tonal_source_mode = st.radio(
    "Source Audio",
    [
        "Current Working Audio",
        "Original Audio",
    ],
    key="tonal_source",
)

run_tonal = st.button(
    "Run Tonal Match",
    key="run_tonal_match",
)

accept_tonal_preview = st.button(
    "Accept",
    key="accept_tonal_preview",
)

if run_tonal:
    source_audio = get_source_audio(tonal_source_mode)
    reference_audio = get_reference_audio()
    current_sr = get_sample_rate()

    if source_audio is None:
        st.error("Please load a source audio before running Tonal Match.")
    elif reference_audio is None:
        st.error("Please load a reference audio before running Tonal Match.")
    elif current_sr is None:
        st.error("Please ensure the working audio has been loaded before processing.")
    else:
        with st.spinner("Running Tonal Match..."):
            try:
                # Run the existing tonal matching routine directly on the selected
                # source and reference audio arrays without altering the DSP logic.
                tonal_output = apply_tonal_matching(
                    reference_audio, source_audio, current_sr
                )

                # Store the tonal preview separately so it can be played, downloaded,
                # and accepted independently from the dynamic-match preview.
                st.session_state["tonal_preview_audio"] = tonal_output
                st.session_state["tonal_preview_sample_rate"] = current_sr

                st.success("Preview Ready")
                st.caption(
                    "The tonal preview was generated without updating the working audio."
                )
            except Exception as exc:
                st.error(f"Tonal Match failed: {exc}")

# ----------------------------------
# Tonal Match Compare Section
# ----------------------------------

st.subheader("🎧 Compare")

compare_tonal_mode = st.radio(
    "Listen To",
    [
        "Original Audio",
        "Tonal Match Preview",
        "Current Working Audio",
    ],
    key="tonal_compare_mode",
    horizontal=True,
)

# Select the single audio source for the compare player so the UI remains simple.
if compare_tonal_mode == "Original Audio":
    compare_audio = get_original_audio()
    compare_sr = get_sample_rate()
elif compare_tonal_mode == "Tonal Match Preview":
    compare_audio = st.session_state.get("tonal_preview_audio")
    compare_sr = st.session_state.get("tonal_preview_sample_rate") or get_sample_rate()
else:
    compare_audio = get_working_audio()
    compare_sr = get_sample_rate()

if st.session_state.get("tonal_preview_audio") is not None:
    preview_wav_bytes = _audio_to_wav_bytes_data(
        st.session_state["tonal_preview_audio"],
        st.session_state.get("tonal_preview_sample_rate") or get_sample_rate(),
    )

    st.download_button(
        label="⬇ Download Preview",
        data=preview_wav_bytes,
        file_name="tonal_preview.wav",
        mime="audio/wav",
        key="download_tonal_preview_wav",
    )

    if accept_tonal_preview:
        set_working_audio(st.session_state["tonal_preview_audio"])
        st.success("Working audio updated from the tonal preview.")

# ----------------------------------
# Tonal Match Analysis Panel
# ----------------------------------

with st.expander("▶ Show EQ Analysis", expanded=False):
    reference_audio = get_reference_audio()
    original_audio = get_original_audio()
    tonal_preview_audio = st.session_state.get("tonal_preview_audio")
    current_sr = get_sample_rate()

    if (
        reference_audio is not None
        and original_audio is not None
        and tonal_preview_audio is not None
        and current_sr
    ):
        st.write("### Frequency Comparison")

        # Compute a smoothed, logarithmic spectrum view that resembles a studio EQ
        # analyzer while keeping the tonal matching DSP untouched.
        def _smoothed_spectrum(audio, sr):
            y = np.asarray(audio, dtype="float32")
            stft = librosa.stft(y, n_fft=4096, hop_length=1024)
            mag = np.abs(stft)
            mean_mag = np.mean(mag, axis=1)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

            # Convert magnitude to dB for a more analyzer-like display.
            mag_db = 20 * np.log10(mean_mag + 1e-9)

            # Apply light smoothing to reduce visual noise and improve readability.
            smooth_window = 9
            if len(mag_db) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                mag_db = np.convolve(mag_db, kernel, mode="same")

            # Restrict to the audible band and keep the low-frequency region visible.
            mask = (freqs >= 20) & (freqs <= 20000)
            freqs = freqs[mask]
            mag_db = mag_db[mask]

            return freqs, mag_db

        ref_freqs, ref_mag_db = _smoothed_spectrum(reference_audio, current_sr)
        orig_freqs, orig_mag_db = _smoothed_spectrum(original_audio, current_sr)
        preview_freqs, preview_mag_db = _smoothed_spectrum(
            tonal_preview_audio, current_sr
        )

        # Align the three spectra onto a shared logarithmic frequency grid while
        # excluding any sub-20 Hz bins that can break log-scale plotting.
        freq_grid = np.geomspace(20, 20000, 400)
        valid_ref = (ref_freqs >= 20) & (ref_freqs <= 20000)
        valid_orig = (orig_freqs >= 20) & (orig_freqs <= 20000)
        valid_preview = (preview_freqs >= 20) & (preview_freqs <= 20000)

        ref_mag_interp = np.interp(
            freq_grid,
            ref_freqs[valid_ref],
            ref_mag_db[valid_ref],
        )
        orig_mag_interp = np.interp(
            freq_grid,
            orig_freqs[valid_orig],
            orig_mag_db[valid_orig],
        )
        preview_mag_interp = np.interp(
            freq_grid,
            preview_freqs[valid_preview],
            preview_mag_db[valid_preview],
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freq_grid,
                y=ref_mag_interp,
                mode="lines",
                name="Reference",
                line=dict(color="#2E8B57", width=2.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freq_grid,
                y=orig_mag_interp,
                mode="lines",
                name="Original",
                line=dict(color="#1F77B4", width=2.0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freq_grid,
                y=preview_mag_interp,
                mode="lines",
                name="Preview",
                line=dict(color="#FF8C00", width=2.0),
            )
        )

        fig.update_layout(
            title="Frequency Comparison",
            xaxis_type="log",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dB)",
            template="plotly_white",
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(
                range=[1.3010, 4.3010],
                tickvals=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                ticktext=[
                    "20",
                    "50",
                    "100",
                    "200",
                    "500",
                    "1k",
                    "2k",
                    "5k",
                    "10k",
                    "20k",
                ],
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Generate a Tonal Match preview to view the EQ comparison chart.")

st.divider()

st.header("🧹 Cleanup")

cleanup_source_mode = st.radio(
    "Source Audio",
    [
        "Current Working Audio",
        "Original Audio",
    ],
    key="cleanup_source",
)

cleanup_options = st.checkbox(
    "Clean Low End",
    value=False,
    key="cleanup_low_end",
)

cleanup_denoise = st.checkbox("Denoise", value=False, key="cleanup_denoise")
cleanup_dereverb = st.checkbox("Dereverb", value=False, key="cleanup_dereverb")
# cleanup_declick = st.checkbox("Declick", value=False, key="cleanup_declick")
cleanup_strip_silence = st.checkbox(
    "Strip Silence",
    value=False,
    key="cleanup_strip_silence",
)

run_cleanup = st.button(
    "Run Cleanup",
    key="run_cleanup",
)

accept_cleanup_preview = st.button(
    "Accept",
    key="accept_cleanup_preview",
)

if run_cleanup:
    source_audio = get_source_audio(cleanup_source_mode)
    current_sr = get_sample_rate()

    if source_audio is None:
        st.error("Please load a source audio before running Cleanup.")
    elif current_sr is None:
        st.error("Please ensure the working audio has been loaded before processing.")
    else:
        with st.spinner("Applying cleanup..."):
            try:
                # Start from the selected source audio and apply only the options
                # requested by the user, keeping the workflow simple and predictable.
                cleaned_audio = source_audio

                if cleanup_options:
                    cleaned_audio = apply_adaptive_hpf(cleaned_audio, current_sr)
                    cleaned_audio = apply_subsonic_cleanup(cleaned_audio, current_sr)

                if cleanup_denoise:
                    cleaned_audio = apply_light_denoise(cleaned_audio, current_sr)

                if cleanup_dereverb:
                    cleaned_audio = apply_light_dereverb(cleaned_audio, current_sr)

                # Temporarily disabled.
                # Current implementation is too weak and requires redesign.

                # if cleanup_declick:
                #     cleaned_audio = apply_light_declick(cleaned_audio, current_sr)

                if cleanup_strip_silence:
                    cleaned_audio = apply_strip_silence(cleaned_audio, current_sr)

                st.session_state["cleanup_preview_audio"] = cleaned_audio
                st.session_state["cleanup_preview_sample_rate"] = current_sr

                st.success("✅ Preview Ready")
                st.caption(
                    "The cleanup preview was generated without updating the working audio."
                )
            except Exception as exc:
                st.error(f"Cleanup failed: {exc}")

st.subheader("🎧 Compare")

compare_cleanup_mode = st.radio(
    "Listen To",
    [
        "Original Audio",
        "Cleanup Preview",
        "Current Working Audio",
    ],
    key="cleanup_compare_mode",
    horizontal=True,
)

# Select the single audio source for the compare player so the UI remains simple.
if compare_cleanup_mode == "Original Audio":
    compare_audio = get_original_audio()
    compare_sr = get_sample_rate()
elif compare_cleanup_mode == "Cleanup Preview":
    compare_audio = st.session_state.get("cleanup_preview_audio")
    compare_sr = (
        st.session_state.get("cleanup_preview_sample_rate") or get_sample_rate()
    )
else:
    compare_audio = get_working_audio()
    compare_sr = get_sample_rate()

if compare_audio is not None:
    st.audio(
        _audio_to_wav_bytes_data(compare_audio, compare_sr),
        format="audio/wav",
    )
else:
    st.info("No audio is available for the selected compare option yet.")

if st.session_state.get("cleanup_preview_audio") is not None:

    preview_wav_bytes = _audio_to_wav_bytes_data(
        st.session_state["cleanup_preview_audio"],
        st.session_state.get("cleanup_preview_sample_rate") or get_sample_rate(),
    )

    st.download_button(
        label="⬇ Download Preview",
        data=preview_wav_bytes,
        file_name="cleanup_preview.wav",
        mime="audio/wav",
        key="download_cleanup_preview_wav",
    )

if accept_cleanup_preview:
    if st.session_state.get("cleanup_preview_audio") is not None:
        set_working_audio(st.session_state["cleanup_preview_audio"])
        st.success("Working audio updated from the cleanup preview.")
    else:
        st.warning("No cleanup preview is available to accept yet.")
