import streamlit as st
import librosa

from core.audio_state import (
    init_audio_state,
    get_working_audio,
    get_reference_audio,
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

init_audio_state()

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
