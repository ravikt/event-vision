import argparse
import os
import cv2
import numpy as np

from detect_prop import init_iterator, visualize_time_surface, detect_circles
from rotation import compute_similarity_for_bboxes, plot_similarity_subplots, estimate_rpm_wavelet
import threading
import math


def detect_initial_bboxes(input_raw, delta_t=10000, frames_to_process=30, min_radius=20, max_radius=120):
    """Build a representative time-surface from the first N chunks and detect circles.

    Returns a list of bboxes [(x_min,y_min,x_max,y_max), ...] or an empty list.
    """
    mv_iterator, width, height = init_iterator(input_raw, delta_t)

    ts_surface = None
    FADE_TIME = 10000

    last_display = None
    count = 0
    for evts in mv_iterator:
        if evts.size == 0:
            continue
        if ts_surface is None:
            ts_surface = np.zeros((height, width), dtype=np.uint64)
        x, y, t = evts['x'], evts['y'], evts['t']
        ts_surface[y, x] = t
        current_time = t[-1]
        age = current_time - ts_surface
        age = np.clip(age, 0, FADE_TIME)
        display_image = (255 * (1.0 - age / FADE_TIME)).astype(np.uint8)
        display_bgr = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        last_display = display_bgr.copy()
        count += 1
        if count >= frames_to_process:
            break

    if last_display is None:
        return []

    _, bboxes = detect_circles(last_display, min_radius=min_radius, max_radius=max_radius)
    # detect_circles returns (image, bboxes)
    return bboxes or []


def main():
    parser = argparse.ArgumentParser(description="Visualize events or compute per-prop similarity signals.")
    parser.add_argument("--input_raw", type=str, required=True, help="Path to input .raw file")
    parser.add_argument("--delta_t", type=int, default=10000, help="Delta time for EventsIterator (in us, default: 10000)")
    parser.add_argument("--compute_similarity", action='store_true', help="Detect propellers and compute per-bbox similarity plots")
    parser.add_argument("--frames", type=int, default=30, help="Initial chunks to build detection frame")
    parser.add_argument("--min_radius", type=int, default=20, help="Hough min radius")
    parser.add_argument("--max_radius", type=int, default=120, help="Hough max radius")
    parser.add_argument("--output", default="similarity_per_prop.png", help="Output PNG filename for similarity subplots")
    args = parser.parse_args()

    if not os.path.exists(args.input_raw):
        print(f"Error: Input file not found: {args.input_raw}")
        return

    if args.compute_similarity:
        # Run similarity computation in background while visualizing the time surface.
        def similarity_job():
            try:
                bboxes = detect_initial_bboxes(args.input_raw, args.delta_t, frames_to_process=args.frames,
                                               min_radius=args.min_radius, max_radius=args.max_radius)
            except Exception as e:
                print(f"Error detecting initial bboxes: {e}")
                return

            if not bboxes:
                print("No circles detected. Try adjusting Hough parameters or increase frames.")
                return

            # Keep up to 4 boxes (largest by area)
            if len(bboxes) > 4:
                areas = [(((x2 - x1) * (y2 - y1)), i) for i, (x1, y1, x2, y2) in enumerate(bboxes)]
                areas.sort(reverse=True)
                keep_idx = [i for _, i in areas[:4]]
                bboxes_local = [bboxes[i] for i in keep_idx]
            else:
                bboxes_local = bboxes

            print(f"Using {len(bboxes_local)} bboxes for similarity computation: {bboxes_local}")

            try:
                time_s, sims = compute_similarity_for_bboxes(args.input_raw, bboxes_local, delta_t=args.delta_t)
            except Exception as e:
                print(f"Error computing similarity: {e}")
                return

            labels = [f"Prop {i+1}" for i in range(len(bboxes_local))]
            try:
                plot_similarity_subplots(time_s, sims, output_plot=args.output, bbox_labels=labels)
            except Exception as e:
                print(f"Error saving similarity plot: {e}")

            # Save CSV
            try:
                out_csv = os.path.splitext(args.output)[0] + '.csv'
                stacked = np.column_stack([time_s] + [s for s in sims])
                header = 'time_s,' + ','.join(labels)
                np.savetxt(out_csv, stacked, delimiter=',', header=header, comments='')
                print(f"Saved similarity CSV to: {out_csv}")
            except Exception as e:
                print(f"Error saving similarity CSV: {e}")

        sim_thread = threading.Thread(target=similarity_job, daemon=True)
        sim_thread.start()

        # Start visualization in main thread so OpenCV display works correctly
        try:
            mv_iterator, width, height = init_iterator(args.input_raw, args.delta_t)
        except Exception as e:
            print(f"Error: {e}")
            return
        visualize_time_surface(mv_iterator, width, height)

        # After visualization window is closed, wait briefly for similarity thread to finish
        if sim_thread.is_alive():
            print("Waiting for similarity computation to finish (up to 5s)...")
            sim_thread.join(timeout=5)
            if sim_thread.is_alive():
                print("Similarity computation still running in background; exiting now.")
    else:
        try:
            mv_iterator, width, height = init_iterator(args.input_raw, args.delta_t)
        except Exception as e:
            print(f"Error: {e}")
            return
        visualize_time_surface(mv_iterator, width, height)


if __name__ == "__main__":
    main()
