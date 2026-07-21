import io

import librosa
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

from engine.dynamic_match import analyze_and_match_vocal

init_audio_state()

if "preview_audio" not in st.session_state:
    st.session_state["preview_audio"] = None

if "preview_sample_rate" not in st.session_state:
    st.session_state["preview_sample_rate"] = None


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

accept_preview = st.button(
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
# Preview Playback & Download
# ----------------------------------

if st.session_state.get("preview_audio") is not None:
    st.subheader("🎧 Preview")

    preview_audio = st.session_state["preview_audio"]
    preview_sr = st.session_state.get("preview_sample_rate") or get_sample_rate()

    # Build WAV bytes once so the same preview can be played back and downloaded.
    preview_wav_bytes = _audio_to_wav_bytes_data(preview_audio, preview_sr)

    st.audio(preview_wav_bytes, format="audio/wav")

    st.download_button(
        label="⬇ Download Preview",
        data=preview_wav_bytes,
        file_name="preview.wav",
        mime="audio/wav",
        key="download_preview_wav",
    )

if accept_preview:
    if st.session_state.get("preview_audio") is not None:
        set_working_audio(st.session_state["preview_audio"])
        st.success("Working audio updated from the preview.")
    else:
        st.warning("No preview is available to accept yet.")
