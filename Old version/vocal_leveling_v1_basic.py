import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import os

def match_vocal_volumes(ref_path, target_path, output_path, smoothness=5):
    print("Načítám audio soubory...")
    # Načtení audia
    y_ref, sr = librosa.load(ref_path, sr=None)
    y_target, _ = librosa.load(target_path, sr=sr)
    
    # Srovnání délky stop
    max_len = max(len(y_ref), len(y_target))
    y_ref = librosa.util.fix_length(y_ref, size=max_len)
    y_target = librosa.util.fix_length(y_target, size=max_len)
    
    print("Analyzuji hlasitostní obálky (RMS)...")
    hop_length = 512
    frame_length = 2048
    
    rms_ref = librosa.feature.rms(y=y_ref, frame_length=frame_length, hop_length=hop_length)[0]
    rms_target = librosa.feature.rms(y=y_target, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Vyhlazení křivek, aby to nechrčelo
    rms_ref_smooth = gaussian_filter1d(rms_ref, sigma=smoothness)
    rms_target_smooth = gaussian_filter1d(rms_target, sigma=smoothness)
    
    print("Aplikuji hlasitostní korekci...")
    epsilon = 1e-5
    gain_curve = (rms_ref_smooth + epsilon) / (rms_target_smooth + epsilon)
    
    # Limitace extrémního zesílení
    gain_curve = np.clip(gain_curve, 0.25, 4.0)
    
    # Natažení křivky na délku audia
    gain_samples = np.interp(
        np.arange(len(y_target)), 
        np.arange(len(gain_curve)) * hop_length, 
        gain_curve
    )
    
    # Úprava hlasitosti
    y_modulated = y_target * gain_samples
    
    print(f"Ukládám výsledný vokál do: {output_path}")
    sf.write(output_path, y_modulated, sr)
    print("✓ Všechno hotovo úspěšně!")

# Spuštění programu v aktuální složce
if __name__ == "__main__":
    match_vocal_volumes("No37-6 english voice.wav", "No37-6 slovak voice.wav", "vysledny_vokal_match.wav")
