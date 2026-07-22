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
st.caption("Listen To")

compare_dynamic_mode = st.radio(
    "",
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

if accept_dynamic_preview:
    if st.session_state.get("preview_audio") is not None:
        set_working_audio(st.session_state["preview_audio"])
        st.success("Working audio updated from the dynamic preview.")
    else:
        st.warning("No dynamic preview is available to accept yet.")

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
st.caption("Listen To")

compare_tonal_mode = st.radio(
    "",
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

if compare_audio is not None:
    st.audio(_audio_to_wav_bytes_data(compare_audio, compare_sr), format="audio/wav")
else:
    st.info("No audio is available for the selected compare option yet.")

if accept_tonal_preview:
    if st.session_state.get("tonal_preview_audio") is not None:
        set_working_audio(st.session_state["tonal_preview_audio"])
        st.success("Working audio updated from the tonal preview.")
    else:
        st.warning("No tonal preview is available to accept yet.")
