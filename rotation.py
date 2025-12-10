import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from metavision_core.event_io import EventsIterator
import os

def compute_similarity_signal(input_path, delta_t=10000):
    """Process .raw file and return time (s) and similarity signal."""
    print(f"Processing {input_path} with delta_t={delta_t} µs...")
    mv_iterator = EventsIterator(input_path=input_path, delta_t=delta_t)
    height, width = mv_iterator.get_size() or (480, 640)

    ts_surface = np.zeros((height, width), dtype=np.uint64)
    FADE_TIME = 10000  # 10 ms

    similarities = []
    timestamps_s = []
    prev_img = None

    for evts in mv_iterator:
        if evts.size == 0:
            continue
        x, y, t = evts['x'], evts['y'], evts['t']
        ts_surface[y, x] = t
        current_time = t[-1]

        age = current_time - ts_surface
        age = np.clip(age, 0, FADE_TIME)
        img = (255 * (1.0 - age / FADE_TIME)).astype(np.uint8)

        if prev_img is not None:
            i1 = prev_img.astype(np.float32)
            i2 = img.astype(np.float32)
            i1 = (i1 - i1.mean()) / (i1.std() + 1e-8)
            i2 = (i2 - i2.mean()) / (i2.std() + 1e-8)
            ncc = float(np.mean(i1 * i2))
            similarities.append(ncc)
            timestamps_s.append(current_time / 1e6)  # seconds

        prev_img = img.copy()

    return np.array(timestamps_s), np.array(similarities)

def estimate_rpm_wavelet(time_s, similarity, num_blades=3, plot=True, output_plot="rpm_wavelet.png"):
    """
    Estimate RPM using Continuous Wavelet Transform (CWT).
    Assumes similarity signal contains periodicity from rotating object.
    """
    if len(similarity) < 20:
        raise ValueError("Signal too short for wavelet analysis")

    # Remove trend (optional but recommended)
    signal = similarity - np.mean(similarity)

    # Sampling rate (from time array)
    dt = np.mean(np.diff(time_s))
    fs = 1.0 / dt  # Hz

    # Define frequency range (e.g., 0.5 Hz to 20 Hz → 30 to 1200 RPM)
    freqs = np.linspace(0.2, 30.0, 100)  # adjust based on expected RPM
    scales = pywt.frequency2scale('morl', freqs) / dt  # morlet wavelet

    # Compute CWT
    coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=dt)

    # Compute power (magnitude squared)
    power = np.abs(coefficients) ** 2

    # Find ridge (most energetic frequency at each time)
    ridge = np.argmax(power, axis=0)
    inst_freq = frequencies[ridge]  # instantaneous frequency in Hz

    # Convert to RPM: freq (Hz) → rotations per second → RPM
    # But: each rotation produces `num_blades` cycles in similarity signal?
    # For now, assume 1 similarity cycle = 1 full rotation → RPM = freq * 60
    rpm = inst_freq * 60.0

    # Optional: smooth RPM
    from scipy.ndimage import median_filter
    rpm_smooth = median_filter(rpm, size=5)

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Original signal
        axs[0].plot(time_s, similarity, 'b-', linewidth=0.8)
        axs[0].set_ylabel("Similarity")
        axs[0].grid(True)

        # Scalogram
        im = axs[1].imshow(power, extent=[time_s[0], time_s[-1], freqs[-1], freqs[0]],
                           cmap='jet', aspect='auto')
        axs[1].set_ylabel("Frequency (Hz)")
        axs[1].set_title("Wavelet Scalogram")
        plt.colorbar(im, ax=axs[1])

        # RPM
        axs[2].plot(time_s, rpm_smooth, 'r-', linewidth=1.2)
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("RPM")
        axs[2].grid(True)
        axs[2].set_ylim(0, np.nanmax(rpm_smooth) * 1.1)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"Saved wavelet RPM plot to: {output_plot}")

    return time_s, rpm_smooth

def main():
    parser = argparse.ArgumentParser(description="Estimate RPM using wavelet transform on time-surface similarity.")
    parser.add_argument("--input", help="Path to .raw or .hdf5 event file")
    parser.add_argument("--delta_t", type=int, default=10000, help="Time slice (µs)")
    parser.add_argument("--num_blades", type=int, default=3, help="Number of blades (for advanced calibration)")
    parser.add_argument("--output_plot", default="rpm_wavelet.png", help="Output plot filename")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Step 1: Compute similarity signal
    time_s, similarity = compute_similarity_signal(args.input, args.delta_t)

    print(f"Generated similarity signal with {len(similarity)} samples over {time_s[-1]:.2f} seconds")

    # Step 2: Estimate RPM via wavelet
    time_rpm, rpm = estimate_rpm_wavelet(
        time_s, similarity,
        num_blades=args.num_blades,
        plot=True,
        output_plot=args.output_plot
    )

    # Optional: save RPM to CSV
    output_csv = args.output_plot.replace('.png', '.csv')
    np.savetxt(output_csv, np.column_stack([time_rpm, rpm]), 
               delimiter=',', header='time_s,rpm', comments='')
    print(f"Saved RPM data to: {output_csv}")

if __name__ == "__main__":
    main()