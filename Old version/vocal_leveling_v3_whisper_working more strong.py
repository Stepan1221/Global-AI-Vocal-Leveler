import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

def match_vocal_volumes_sweet_spot(ref_path, target_path, output_path, smoothness=65):
    print("Načítám audio soubory...")
    y_ref, sr = librosa.load(ref_path, sr=None)
    y_target, _ = librosa.load(target_path, sr=sr)
    
    # Srovnání délek stop
    max_len = max(len(y_ref), len(y_target))
    y_ref = librosa.util.fix_length(y_ref, size=max_len)
    y_target = librosa.util.fix_length(y_target, size=max_len)
    
    # Výpočet celkové hlasitosti
    rms_global_ref = np.sqrt(np.mean(y_ref**2))
    rms_global_target = np.sqrt(np.mean(y_target**2))
    
    # Základní faktor vyrovnání + kompenzace 1.2 dB
    global_gain_factor = rms_global_ref / (rms_global_target + 1e-6)
    fine_tune_factor = 10**(1.2 / 20)
    final_global_gain = global_gain_factor * fine_tune_factor
    
    # Normalizace pro vnitřní výpočet křivky
    y_ref_norm = y_ref / (np.max(np.abs(y_ref)) + 1e-6)
    y_target_norm = y_target / (np.max(np.abs(y_target)) + 1e-6)
    
    print("Analýza detailní hlasitosti (RMS)...")
    hop_length = 512
    frame_length = 2048
    
    rms_ref = librosa.feature.rms(y=y_ref_norm, frame_length=frame_length, hop_length=hop_length)[0]
    rms_target = librosa.feature.rms(y=y_target_norm, frame_length=frame_length, hop_length=hop_length)[0]
    
    rms_ref_smooth = gaussian_filter1d(rms_ref, sigma=smoothness)
    rms_target_smooth = gaussian_filter1d(rms_target, sigma=smoothness)
    
    print("Výpočet mikro-korekce...")
    epsilon = 1e-3
    gain_curve = (rms_ref_smooth + epsilon) / (rms_target_smooth + epsilon)
    
    # ZLATÁ STŘEDNÍ CESTA: Limit stažen z 0.25/2.5 na rozumných 0.35 až 2.0
    gain_curve = np.clip(gain_curve, 0.35, 2.0)
    gain_curve = gaussian_filter1d(gain_curve, sigma=smoothness)
    
    # Roztažení křivky na samply
    gain_samples = np.interp(
        np.arange(len(y_target)), 
        np.arange(len(gain_curve)) * hop_length, 
        gain_curve
    )
    
    # Aplikace úpravy
    y_modulated = y_target * gain_samples * final_global_gain
    
    # Ochrana proti clippingu
    max_val = np.max(np.abs(y_modulated))
    if max_val > 0.99:
        y_modulated = y_modulated / max_val * 0.99
    
    print(f"Ukládám vybalancovaný vokál do: {output_path}")
    sf.write(output_path, y_modulated, sr)
    print("✓ Hotovo! Zlatý střed je vyexportován.")

if __name__ == "__main__":
    match_vocal_volumes_sweet_spot("No37-6 english voice without breath.wav", "No37-6 slovak voice.wav", "vysledny_vokal_pure_match.wav")
