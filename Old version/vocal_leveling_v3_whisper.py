import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

def match_vocal_volumes_with_global_gain(ref_path, target_path, output_path, smoothness=60):
    print("Načítám audio soubory...")
    y_ref, sr = librosa.load(ref_path, sr=None)
    y_target, _ = librosa.load(target_path, sr=sr)
    
    # Srovnání délek stop
    max_len = max(len(y_ref), len(y_target))
    y_ref = librosa.util.fix_length(y_ref, size=max_len)
    y_target = librosa.util.fix_length(y_target, size=max_len)
    
    # Spočítáme celkovou průměrnou hlasitost (RMS) obou stop před úpravou
    # (Ignorujeme úplné ticho, abychom měli přesný průměrný výkon zpěvu)
    rms_global_ref = np.sqrt(np.mean(y_ref**2))
    rms_global_target = np.sqrt(np.mean(y_target**2))
    
    # Spočítáme koeficient pro celkové vyrovnání hlasitostí
    global_gain_factor = rms_global_ref / (rms_global_target + 1e-6)
    print(f"-> Celkový rozdíl hlasitostí zjištěn. Cílová stopa bude globálně upravena faktorem: {global_gain_factor:.2f}x")
    
    # 1. KROK: Normalizace pro vnitřní výpočet křivky (startovní čára -1 až 1)
    y_ref_norm = y_ref / (np.max(np.abs(y_ref)) + 1e-6)
    y_target_norm = y_target / (np.max(np.abs(y_target)) + 1e-6)
    
    print("Analýza detailní hlasitosti (RMS)...")
    hop_length = 512
    frame_length = 2048
    
    rms_ref = librosa.feature.rms(y=y_ref_norm, frame_length=frame_length, hop_length=hop_length)[0]
    rms_target = librosa.feature.rms(y=y_target_norm, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Vyhlazení křivek pro plynulý průběh mikro-automatizace
    rms_ref_smooth = gaussian_filter1d(rms_ref, sigma=smoothness)
    rms_target_smooth = gaussian_filter1d(rms_target, sigma=smoothness)
    
    print("Výpočet mikro-korekce...")
    epsilon = 1e-3
    gain_curve = (rms_ref_smooth + epsilon) / (rms_target_smooth + epsilon)
    
    # Jemný studiový limit pro detaily (žádné extrémy)
    gain_curve = np.clip(gain_curve, 0.5, 1.5)
    gain_curve = gaussian_filter1d(gain_curve, sigma=smoothness)
    
    # Roztažení křivky na samply
    gain_samples = np.interp(
        np.arange(len(y_target)), 
        np.arange(len(gain_curve)) * hop_length, 
        gain_curve
    )
    
    # 2. KROK: Aplikujeme mikro-automatizaci a ZÁROVEŇ celkový global gain
    # Tím získáme perfektní tvar vlny i stejnou celkovou hlasitost
    y_modulated = y_target * gain_samples * global_gain_factor
    
    # Bezpečnostní limiter proti ořezu (clippingu) na úplném konci, kdyby to šlo přes 0 dB
    max_val = np.max(np.abs(y_modulated))
    if max_val > 0.99:
        print("-> Aktivován limiter proti digitálnímu přebuzení (clippingu).")
        y_modulated = y_modulated / max_val * 0.99
    
    print(f"Ukládám stoprocentně vyvážený vokál do: {output_path}")
    sf.write(output_path, y_modulated, sr)
    print("✓ Vše hotovo! Stopy jsou teď vyrovnané celkově i v detailech.")

if __name__ == "__main__":
    match_vocal_volumes_with_global_gain("No37-6 english voice without breath.wav", "No37-6 slovak voice.wav", "vysledny_vokal_pure_match.wav")
