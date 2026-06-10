import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import pandas as pd
import mido
from mido import MidiFile, MidiTrack, Message

def calculate_metrics(y, sr):
    # Helper function to calculate audio industry metrics in dB
    rms_val = np.sqrt(np.mean(y**2))
    rms_db = 20 * np.log10(rms_val + 1e-6)
    
    peak_val = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak_val + 1e-6)
    
    # Dynamic Range as Crest Factor (Peak to RMS)
    dynamic_range = peak_db - rms_db
    
    return {
        "Volume (RMS)": f"{rms_db:.1f} dB",
        "Dynamic Range": f"{dynamic_range:.1f} dB",
        "Peak (Maximum)": f"{peak_db:.1f} dB"
    }

def analyze_and_match_vocal(ref_file, target_file, fader_speed="Normal", intensity=70, output_trim=1.2, auto_mode=True):
    # 1. Load Audio Files
    y_ref, sr = librosa.load(ref_file, sr=None)
    y_target, _ = librosa.load(target_file, sr=sr)
    
    max_len = max(len(y_ref), len(y_target))
    y_ref = librosa.util.fix_length(y_ref, size=max_len)
    y_target = librosa.util.fix_length(y_target, size=max_len)
    
    hop_length = 512
    frame_length = 2048
    
    # Root Mean Square (RMS) Envelopes
    y_ref_norm = y_ref / (np.max(np.abs(y_ref)) + 1e-6)
    y_target_norm = y_target / (np.max(np.abs(y_target)) + 1e-6)
    
    rms_ref = librosa.feature.rms(y=y_ref_norm, frame_length=frame_length, hop_length=hop_length)[0]
    rms_target = librosa.feature.rms(y=y_target_norm, frame_length=frame_length, hop_length=hop_length)[0]
    
    num_frames = len(rms_ref)
    
    # --- SMART AUTO-ANALYZE ENGINE ---
    if auto_mode:
        try:
            y_harm, y_perc = librosa.effects.hpss(y_ref)
            rms_perc = librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0]
            rms_harm = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=hop_length)[0]
            
            percussive_ratio = (rms_perc + 1e-6) / (rms_harm + rms_perc + 1e-6)
            spec_centroid = librosa.feature.spectral_centroid(y=y_ref, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
            spec_norm = np.clip(spec_centroid / 4000.0, 0.0, 1.0)
            
            dynamic_intensity = 0.65 + (percussive_ratio * 0.15) - (spec_norm * 0.08)
            dynamic_intensity = np.clip(dynamic_intensity, 0.55, 0.75) 
            
            dynamic_sigma = 75.0 - (percussive_ratio * 40.0)
            dynamic_sigma = np.clip(dynamic_sigma, 40.0, 95.0)
            
            intensity_curve = gaussian_filter1d(dynamic_intensity, sigma=50)
            sigma_curve = gaussian_filter1d(dynamic_sigma, sigma=50)
            
            intensity = int(np.mean(intensity_curve) * 100)
            avg_sigma = np.mean(sigma_curve)
            if avg_sigma < 55: fader_speed = "Dynamic: Fast / Sharp"
            elif avg_sigma > 75: fader_speed = "Dynamic: Smooth / Loose"
            else: fader_speed = "Dynamic: Balanced Center"
            
        except:
            intensity_curve = np.full(num_frames, 0.68)
            sigma_curve = np.full(num_frames, 65.0)
            intensity = 68
            fader_speed = "Normal (Backup)"
    else:
        if fader_speed == "Fast (Sharp)":
            static_sigma = 45.0
        elif fader_speed == "Normal (Medium)":
            static_sigma = 65.0
        elif fader_speed == "Slow (Loose)":
            static_sigma = 95.0
            
        intensity_curve = np.full(num_frames, intensity / 100.0)
        sigma_curve = np.full(num_frames, static_sigma)

    # Envelope Smoothing
    if auto_mode:
        rms_ref_smooth = np.zeros_like(rms_ref)
        rms_target_smooth = np.zeros_like(rms_target)
        for i in range(num_frames):
            current_sigma = sigma_curve[i]
            rms_ref_smooth[i] = gaussian_filter1d(rms_ref, sigma=current_sigma)[i]
            rms_target_smooth[i] = gaussian_filter1d(rms_target, sigma=current_sigma)[i]
    else:
        rms_ref_smooth = gaussian_filter1d(rms_ref, sigma=sigma_curve[0])
        rms_target_smooth = gaussian_filter1d(rms_target, sigma=sigma_curve[0])
    
    # Global Energy Match
    rms_global_ref = np.sqrt(np.mean(y_ref**2))
    rms_global_target = np.sqrt(np.mean(y_target**2))
    
    global_gain_factor = rms_global_ref / (rms_global_target + 1e-6)
    fine_tune_factor = 10**(output_trim / 20)
    final_global_gain = global_gain_factor * fine_tune_factor
    
    # Calculate Gain Curve Modulation
    epsilon = 1e-3
    pure_gain_curve = np.clip((rms_ref_smooth + epsilon) / (rms_target_smooth + epsilon), 0.45, 1.65) 
    gain_curve = 1.0 + intensity_curve * (pure_gain_curve - 1.0)
    
    if auto_mode:
        gain_curve = gaussian_filter1d(gain_curve, sigma=int(np.mean(sigma_curve)))
    else:
        gain_curve = gaussian_filter1d(gain_curve, sigma=sigma_curve[0])
    
    gain_samples = np.interp(
        np.arange(len(y_target)), 
        np.arange(len(gain_curve)) * hop_length, 
        gain_curve
    )
    
    # A. Process Target Audio (Scenario A: Ideal)
    y_modulated = y_target * gain_samples * final_global_gain
    
    # No-Gate Safety Logic for Digital Silence
    silence_threshold = 0.005
    rms_ref_samples = np.interp(
        np.arange(len(y_ref)),
        np.arange(len(rms_ref_smooth)) * hop_length,
        rms_ref_smooth
    )
    y_modulated[rms_ref_samples < silence_threshold] = 0
    
    max_val = np.max(np.abs(y_modulated))
    if max_val > 0.99:
        y_modulated = y_modulated / max_val * 0.99
        
    # B. Generate MIDI Automation File (Scenario B: Backup / Problem Fix)
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    time_per_frame_sec = hop_length / sr
    ticks_per_beat = 480
    bpm = 120  
    ticks_per_sec = (ticks_per_beat * bpm) / 60
    ticks_per_frame = int(time_per_frame_sec * ticks_per_sec)
    
    last_cc_val = -1
    for i, g_val in enumerate(gain_curve):
        cc_val = int(np.clip((g_val - 0.45) / (1.65 - 0.45) * 127, 0, 127))
        
        if cc_val != last_cc_val:
            # CC #7 is the worldwide MIDI standard for Volume Automation
            track.append(Message('control_change', control=7, value=cc_val, time=ticks_per_frame if i > 0 else 0))
            last_cc_val = cc_val
        else:
            if i > 0 and len(track) > 0:
                track[-1].time += ticks_per_frame

    times = librosa.times_like(rms_ref_smooth, sr=sr, hop_length=hop_length)
    
    metrics_ref = calculate_metrics(y_ref, sr)
    metrics_target = calculate_metrics(y_target, sr)
    metrics_out = calculate_metrics(y_modulated, sr)
    
    return y_modulated, mid, sr, times, rms_ref_smooth, rms_target_smooth, gain_curve, fader_speed, intensity, metrics_ref, metrics_target, metrics_out

# --- WEB INTERFACE ---
st.set_page_config(page_title="AI Vocal Leveler", page_icon="🎤", layout="centered")

st.title("🎤 AI Vocal Leveler")
st.subheader("Automated Volume Dynamics Matching")
st.write("Upload the reference track and your target language track to automatically match the volume dynamics.")

ref_upload = st.file_uploader("1. Upload Reference Vocal (e.g., English WAV)", type=["wav"])
target_upload = st.file_uploader("2. Upload Target Vocal (e.g., Your Language WAV)", type=["wav"])

st.write("---")
st.subheader("🎛️ Control Panel")

auto_mode = st.toggle("🧠 Smart Auto-Analyze (Recommended)", value=True)

if auto_mode:
    st.info("💡 **Smart Auto-Mode is ACTIVE.** The system automatically analyzes the song structure and dynamically adjusts fader speed and match intensity.")
    fader_speed = "Auto"
    intensity = 70
    output_trim = 1.2
else:
    st.warning("🎚️ **Manual Control Mode Active.**")
    col1, col2 = st.columns(2)
    with col1:
        fader_speed = st.select_slider("Fader Speed (Reaction)", options=["Slow (Loose)", "Normal (Medium)", "Fast (Sharp)"], value="Normal (Medium)")
    with col2:
        intensity = st.slider("Match Intensity (Aggressiveness %)", min_value=10, max_value=100, value=70, step=5)
    
    output_trim = st.slider("Output Trim (Fine-tune Gain in dB)", min_value=-3.0, max_value=3.0, value=1.2, step=0.1)

if ref_upload and target_upload:
    if st.button("⚡ Process and Match Volumes", type="primary"):
        with st.spinner("Analyzing human factor dynamics and processing audio..."):
            try:
                output_audio, midi_data, sample_rate, times, rms_ref, rms_target, gain_curve, final_speed, final_intensity, m_ref, m_tgt, m_out = analyze_and_match_vocal(
                    ref_upload, target_upload, fader_speed, intensity, output_trim, auto_mode
                )
                
                output_fn = "leveled_target_vocal.wav"
                midi_fn = "vocal_volume_automation.mid"
                
                sf.write(output_fn, output_audio, sample_rate)
                midi_data.save(midi_fn)
                
                st.success("✓ Audio successfully leveled!")
                
                if auto_mode:
                    st.code(f"AI Song Analysis Completed:\n -> Automatically selected fader mode: {final_speed}\n -> Calculated safe match intensity: {final_intensity}%")
                
                # TECHNICAL METRICS TABLE
                st.subheader("📊 Technical Audio Metrics")
                data_metrics = {
                    "Metric": ["Overall Volume (RMS)", "Dynamic Range (Crest)", "Peak Volume (Maximum)"],
                    "1. Reference (Original)": [m_ref["Volume (RMS)"], m_ref["Dynamic Range"], m_ref["Peak (Maximum)"]],
                    "2. Target (Before Fix)": [m_tgt["Volume (RMS)"], m_tgt["Dynamic Range"], m_tgt["Peak (Maximum)"]],
                    "3. Output (After AI Fix)": [m_out["Volume (RMS)"], m_out["Dynamic Range"], m_out["Peak (Maximum)"]]
                }
                df = pd.DataFrame(data_metrics)
                st.table(df)
                
                # PLOT GRAPH
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                fig.patch.set_facecolor('#0e1117')
                
                ax1.set_facecolor('#131722')
                ax1.plot(times, rms_ref, label="Reference Envelope (Destination)", color="#f39c12", linewidth=2)
                ax1.plot(times, rms_target, label="Original Target Envelope", color="#3498db", linewidth=1.5, linestyle="--")
                ax1.set_title("Volume Envelopes (RMS) Comparison", color="white", fontsize=12)
                ax1.legend(loc="upper right")
                ax1.grid(True, color="#2c3e50", linestyle=":")
                ax1.tick_params(colors='white')
                
                ax2.set_facecolor('#131722')
                ax2.plot(times, gain_curve, label="Applied Gain Automation", color="#2ecc71", linewidth=2)
                ax2.axhline(1.0, color="white", linestyle=":", alpha=0.5)
                ax2.set_title("Applied Gain Automation Curve (Variable in Time)", color="white", fontsize=12)
                ax2.set_xlabel("Time (seconds)", color="white")
                ax2.set_ylabel("Gain Factor", color="white")
                ax2.legend(loc="upper right")
                ax2.grid(True, color="#2c3e50", linestyle=":")
                ax2.tick_params(colors='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # DOWNLOAD SECTION (Scenario A & Scenario B)
                st.write("---")
                st.subheader("💾 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("🟢 **Scenario A: Ideal Output**")
                    st.caption("Pre-rendered and fully leveled WAV file ready for the mix.")
                    st.audio(output_fn, format="audio/wav")
                    with open(output_fn, "rb") as file:
                        st.download_button(
                            label="🚀 Download Leveled Vocal WAV",
                            data=file,
                            file_name="leveled_target_vocal.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                        
                with col2:
                    st.write("🟡 **Scenario B: Troubleshooting / Backup**")
                    st.caption("Pure automation curve data. Drag & Drop this MIDI file onto your unedited vocal track volume lane in Pro Tools.")
                    with open(midi_fn, "rb") as file:
                        st.download_button(
                            label="🎛️ Download MIDI Automation Curve",
                            data=file,
                            file_name="vocal_volume_automation.mid",
                            mime="audio/midi",
                            use_container_width=True
                        )
                
                os.remove(output_fn)
                os.remove(midi_fn)
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
