# AI Vocal Leveler

This project contains a Streamlit app for automatic vocal level matching. The app compares a reference vocal track with a target vocal track and adjusts the target volume to match the reference.

## Main features

- `Smart Auto-Mode`
  - recommended mode for users who want a simple, automatic setup
  - works automatically with an adaptive model that follows the song over time
  - basic controls are automatic; manual tuning is available in the advanced section

- `Smoothing Mode`
  - options: `Smooth`, `Balanced`, `Sharp`
  - `Smooth` = gentler response, smoother volume changes
  - `Balanced` = a natural compromise between stability and reactivity
  - `Sharp` = faster reaction to short syllables and transients

- `Match Intensity (Aggressiveness)`
  - controls how strong the volume matching is
  - left = more natural, smaller changes
  - right = stronger matching to the reference

- `Onset Sensitivity`
  - controls how much the app reacts to fast transients and word starts
  - low value = smoother, less aggressive changes
  - high value = faster reaction to short syllables and attacks

- `Output Trim`
  - fine-tunes the final output level in dB

- `Advanced settings` are hidden in an expandable section
  - this keeps the main UI simple for users in auto mode
  - manual options are available only when expanded

## Technical improvements

- gain matching is calculated in dB for more natural results
- adaptive attack/release reacts to onset strength
- dynamic weighting between phrase and micro-transient correction
- hysteresis gating preserves breaths and soft endings
- clearer UI with an advanced settings section

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Note

This file is meant to be shared with other developers and technicians so they can understand the current app features and behavior.
