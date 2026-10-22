import numpy as np
import pyloudnorm as pyln


def calculate_r128_metrics(y, sr):
    try:
        y = y.astype("float32")

        meter = pyln.Meter(sr)

        loudness = meter.integrated_loudness(y)
        loudness_range = meter.loudness_range(y)

        # --- Approximate True Peak (oversampling x4) ---
        upsample_factor = 4
        y_upsampled = np.interp(
            np.linspace(0, len(y), len(y) * upsample_factor), np.arange(len(y)), y
        )

        peak_val = np.max(np.abs(y_upsampled))
        true_peak = 20 * np.log10(peak_val + 1e-9)

        return {
            "Integrated Loudness": f"{loudness:.1f} LUFS",
            "Loudness Range": f"{loudness_range:.1f} LU",
            "True Peak": f"{true_peak:.1f} dBTP",
        }

    except Exception:
        return {
            "Integrated Loudness": "Error",
            "Loudness Range": "Error",
            "True Peak": "Error",
        }
