import streamlit as st


def init_audio_state():

    if "original_audio" not in st.session_state:
        st.session_state["original_audio"] = None

    if "working_audio" not in st.session_state:
        st.session_state["working_audio"] = None

    if "reference_audio" not in st.session_state:
        st.session_state["reference_audio"] = None

    if "sample_rate" not in st.session_state:
        st.session_state["sample_rate"] = None


def set_original_audio(audio):

    st.session_state["original_audio"] = audio


def get_original_audio():

    return st.session_state["original_audio"]


def set_working_audio(audio):

    st.session_state["working_audio"] = audio


def get_working_audio():

    return st.session_state["working_audio"]


def set_reference_audio(audio):

    st.session_state["reference_audio"] = audio


def get_reference_audio():

    return st.session_state["reference_audio"]


def set_sample_rate(sr):

    st.session_state["sample_rate"] = sr


def get_sample_rate():

    return st.session_state["sample_rate"]


def has_original_audio():

    return st.session_state["original_audio"] is not None


def has_working_audio():

    return st.session_state["working_audio"] is not None


def has_reference_audio():

    return st.session_state["reference_audio"] is not None


def get_source_audio(source_mode):

    if source_mode == "Original Audio":
        return get_original_audio()

    return get_working_audio()
