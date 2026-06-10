# AI Vocal Leveler

A Streamlit app for automatic vocal dynamics matching. Upload a reference vocal track and a target vocal track, and the app adjusts the target vocal's volume envelope to match the reference.

## Main features

- Upload two WAV files: reference vocal and localized/target vocal
- Automatic vocal level matching with adaptive gain automation
- Optional advanced controls available in an expandable panel

## Controls

- `Smoothing Mode`
  - `Smooth` = gentler processing, more natural long-form smoothing
  - `Balanced` = medium response for natural vocal flow
  - `Sharp` = faster adaptation for quick syllables and transients

- `Match Intensity`
  - adjusts how strongly the app corrects the target vocal
  - lower values keep the original feel, higher values apply stronger leveling

- `Onset Sensitivity`
  - controls reaction speed to transient attacks and short syllables
  - lower values = smoother, less reactive output
  - higher values = more aggressive transient tracking

## Output

- Downloads a leveled WAV file named `leveled_target_vocal.wav`
- Shows loudness metrics for:
  - reference vocal
  - original target vocal
  - processed output vocal
- Displays RMS envelope comparison and applied gain automation curve

## Technical details

- Uses dB-domain gain matching for more natural loudness correction
- Applies phrase-level and micro-level smoothing with adaptive weighting
- Onset-based sensitivity helps preserve transient clarity
- Hysteresis gating avoids pumping during breaths and quiet passages
- Global energy rebalance keeps output aligned with the reference

## Project files

- `app.py` — hlavní Streamlit aplikace pro nahrání, analýzu a zpracování vokálů
- `requirements.txt` — potřebné Python balíčky
- `README.md` — popis a instrukce pro spuštění projektu

## Requirements

- Python 3.9+
- `streamlit`
- `numpy`
- `librosa`
- `soundfile`
- `scipy`
- `matplotlib`
- `pandas`

Install with:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Notes

- Use WAV files for best compatibility.
- The app is designed for offline processing of vocal stems and export of a leveled WAV file.
