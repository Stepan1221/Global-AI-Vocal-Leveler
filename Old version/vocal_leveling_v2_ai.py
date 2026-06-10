import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from fastdtw import fastdtw
from scipy.spatial.distance import cdist

def match_vocals_with_ai(ref_path, target_path, output_path, smoothness=5):
    print("1. Načítám audio soubory do AI paměti...")
    y_ref, sr = librosa.load(ref_path, sr=None)
    y_target, _ = librosa.load(target_path, sr=sr)
    
    print("2. Analyzuji hlasitostní obálky (RMS)...")
    hop_length = 512
    frame_length = 2048
    
    rms_ref = librosa.feature.rms(y=y_ref, frame_length=frame_length, hop_length=hop_length)[0]
    rms_target = librosa.feature.rms(y=y_target, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Vyhlazení pro přirozenější průběh
    rms_ref_smooth = gaussian_filter1d(rms_ref, sigma=smoothness)
    rms_target_smooth = gaussian_filter1d(rms_target, sigma=smoothness)
    
    print("3. Spouštím inteligenci: Hledám shodu frázování a slov (DTW)...")
    # Příprava dat pro porovnání (musí mít správný matematický tvar)
    ref_data = rms_ref_smooth.reshape(-1, 1)
    target_data = rms_target_smooth.reshape(-1, 1)
    
    # Algoritmus najde nejlepší cestu (shodu) mezi frázemi v obou jazycích
    distance, path = fastdtw(ref_data, target_data, dist=2)
    
    print("4. Přepočítávám a ohýbám hlasitostní křivku podle reference...")
    # Vytvoříme novou prázdnou křivku pro cílový vokál
    aligned_gain_curve = np.ones_like(rms_target_smooth)
    
    epsilon = 1e-5
    # Projdeme mapu shody, kterou algoritmus našel, a přeneseme poměry hlasitostí
    for ref_idx, target_idx in path:
        ref_val = rms_ref_smooth[ref_idx]
        target_val = rms_target_smooth[target_idx]
        
        # Spočítáme potřebný gain (zisk) v daném slově/slabice
        gain = (ref_val + epsilon) / (target_val + epsilon)
        aligned_gain_curve[target_idx] = gain

    # Ochrana před extrémním řevem nebo totálním tichem (max 4x zesílení / zeslabení)
    aligned_gain_curve = np.clip(aligned_gain_curve, 0.25, 4.0)
    
    # Vyhladíme výslednou křivku zisku, aby to v přechodech neklikalo
    final_gain_smooth = gaussian_filter1d(aligned_gain_curve, sigma=smoothness)
    
    print("5. Aplikuji mikro-automatizaci na jednotlivé samply...")
    # Natažení frame křivky na délku celého audia
    gain_samples = np.interp(
        np.arange(len(y_target)), 
        np.arange(len(final_gain_smooth)) * hop_length, 
        final_gain_smooth
    )
    
    # Finální úprava cílového zpěvu
    y_modulated = y_target * gain_samples
    
    print(f"6. Ukládám inteligentně vyvážený zpěv do: {output_path}")
    sf.write(output_path, y_modulated, sr)
    print("✓ AI úprava úspěšně dokončena!")

# Spuštění programu
if __name__ == "__main__":
    # Tady nechej přesně ty názvy svých souborů, jako minule
    match_vocals_with_ai("No37-6 english voice.wav", "No37-6 slovak voice.wav", "vysledny_vokal_match.wav")
