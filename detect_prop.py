import numpy as np
import cv2
import os
from metavision_core.event_io import EventsIterator
from rotation import estimate_rpm_wavelet
import time as _time

# --- Configuration ---
# Default resolution for a common event camera (Prophesee VGA)
# Adjust if your data has a different resolution
WIDTH, HEIGHT = 1280, 720

__all__ = [
    "init_iterator",
    "detect_circles",
    "visualize_time_surface",
    "WIDTH",
    "HEIGHT",
]


def init_iterator(input_raw: str, delta_t: int = 10000):
    """
    Initialize a Metavision `EventsIterator` from a file path.

    Returns: (mv_iterator, width, height)
    Raises FileNotFoundError or RuntimeError on failure.
    """
    raw_path = os.path.expanduser(input_raw)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Input file not found at {raw_path}")

    try:
        mv_iterator = EventsIterator(input_path=raw_path, delta_t=delta_t)
    except Exception as e:
        raise RuntimeError(f"Error initializing EventsIterator: {e}")

    # Get sensor size from iterator metadata (if available)
    try:
        size = mv_iterator.get_size()
        if size is not None:
            # `get_size()` returns (height, width)
            height_from_meta, width_from_meta = size
            return mv_iterator, width_from_meta, height_from_meta
    except Exception:
        # Fallback to defaults if metadata retrieval fails
        pass

    return mv_iterator, WIDTH, HEIGHT

# --- Circle Detection Function ---
def detect_circles(image, min_radius=30, max_radius=100, param1=50, param2=30):
    """
    Detect circles in the image and draw a bounding box for each detected circle.

    Parameters:
        image (ndarray): The input BGR image.
        min_radius (int): Minimum circle radius to detect.
        max_radius (int): Maximum circle radius to detect.
        param1 (int): Higher threshold for the internal Canny edge detector.
        param2 (int): Accumulator threshold for center detection.

    Returns:
        result_image (ndarray): Image with bounding boxes drawn (if any).
        bboxes (list): List of (x_min, y_min, x_max, y_max) tuples, one per detected circle.
    """
    # Convert to grayscale and denoise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.medianBlur(gray, 5)
    gray_blurred = cv2.GaussianBlur(gray_denoised, (15, 15), 0)

    # Detect circles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return image, []

    circles = np.uint16(np.around(circles))[0]

    bboxes = []
    for c in circles:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])

        x_min = cx - r
        y_min = cy - r
        x_max = cx + r
        y_max = cy + r

        # Add a small padding (5% of width/height) for visualization
        pad_x = max(5, int(0.05 * (x_max - x_min)))
        pad_y = max(5, int(0.05 * (y_max - y_min)))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(image.shape[1] - 1, x_max + pad_x)
        y_max = min(image.shape[0] - 1, y_max + pad_y)

        # Draw bounding box (red, 2 px)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        bboxes.append((x_min, y_min, x_max, y_max))

    return image, bboxes


# --- Visualization Function ---
def visualize_time_surface(mv_iterator, width, height):
    """
    Iterates through events and visualizes them as a Time Surface.

    This version also runs a simple centroid-based tracker and computes per-track
    similarity (NCC) between consecutive time-surface frames inside the track bbox.
    When enough samples have been collected the code estimates RPM using
    `estimate_rpm_wavelet` and overlays the RPM near the corresponding bbox.
    """
    print(f"Starting visualization for {width}x{height} resolution.")
    print("Press 'SPACE' to pause/resume. Press 'q' or 'ESC' to quit.")

    ts_surface = np.zeros((height, width), dtype=np.uint64)
    display_image = np.zeros((height, width), dtype=np.uint8)
    FADE_TIME = 10000

    paused = False

    # Simple track structure
    tracks = {}  # id -> dict with keys: bbox, centroid, last_seen, sim_list, time_list, rpm
    next_track_id = 1
    MAX_TRACK_LOST = 5  # frames until a track is removed
    MATCH_DIST = 80  # pixels max centroid distance for matching

    prev_img = None
    frame_idx = 0

    for evts in mv_iterator:
        if evts.size == 0:
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused

            while paused:
                key = cv2.waitKey(100)
                if key == ord(' '):
                    paused = not paused
                    break
                elif key == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    return
            continue

        x = evts['x']
        y = evts['y']
        t = evts['t']
        ts_surface[y, x] = t
        current_time = t[-1]

        time_diff = current_time - ts_surface
        clipped_diff = np.clip(time_diff, 0, FADE_TIME)
        intensity = 255 * (1.0 - clipped_diff / FADE_TIME)
        display_image = intensity.astype(np.uint8)
        display_bgr = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        # Detect circles this frame
        result_image, detected_bboxes = detect_circles(display_bgr)

        # Convert bboxes to list of (x_min,y_min,x_max,y_max) if detect_circles returned circles earlier
        detections = detected_bboxes

        # Compute centroids for detections
        det_centroids = []
        for bbox in detections:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            det_centroids.append((cx, cy))

        # Matching detections to existing tracks by nearest centroid
        unmatched_dets = set(range(len(detections)))
        matched = {}

        # Build list of track centroids
        track_items = list(tracks.items())
        for tid, track in track_items:
            tx, ty = track['centroid']
            best_det = None
            best_dist = float('inf')
            for di in list(unmatched_dets):
                cx, cy = det_centroids[di]
                d = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_det = di
            if best_det is not None and best_dist <= MATCH_DIST:
                # match
                matched[tid] = best_det
                unmatched_dets.remove(best_det)
                tracks[tid]['last_seen'] = frame_idx
            else:
                # track not matched this frame
                pass

        # Create new tracks for unmatched detections
        for di in list(unmatched_dets):
            bbox = detections[di]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tracks[next_track_id] = {
                'bbox': bbox,
                'centroid': (cx, cy),
                'last_seen': frame_idx,
                'sim_list': [],
                'time_list': [],
                'rpm': float('nan'),
            }
            matched[next_track_id] = di
            next_track_id += 1

        # Compute similarity per matched track (using prev_img and current display_image)
        if prev_img is not None:
            for tid, di in matched.items():
                det_bbox = detections[di]
                x1, y1, x2, y2 = [int(v) for v in det_bbox]

                # prev_crop from prev_img using previous track bbox if available, else use current bbox
                prev_bbox = tracks[tid]['bbox'] if 'bbox' in tracks[tid] else det_bbox
                px1, py1, px2, py2 = [int(v) for v in prev_bbox]

                # Clamp
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(prev_img.shape[1] - 1, px2), min(prev_img.shape[0] - 1, py2)
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(display_image.shape[1] - 1, x2), min(display_image.shape[0] - 1, y2)

                prev_crop = prev_img[py1:py2 + 1, px1:px2 + 1].astype(np.float32) if (px2 >= px1 and py2 >= py1) else None
                curr_crop = display_image[y1c:y2c + 1, x1c:x2c + 1].astype(np.float32) if (x2c >= x1c and y2c >= y1c) else None

                if prev_crop is None or curr_crop is None or prev_crop.size == 0 or curr_crop.size == 0:
                    tracks[tid]['sim_list'].append(np.nan)
                    tracks[tid]['time_list'].append(current_time / 1e6)
                    continue

                # Resize prev_crop to curr_crop shape if needed
                if prev_crop.shape != curr_crop.shape:
                    try:
                        curr_h, curr_w = curr_crop.shape
                        prev_crop = cv2.resize(prev_crop, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    except Exception:
                        tracks[tid]['sim_list'].append(np.nan)
                        tracks[tid]['time_list'].append(current_time / 1e6)
                        continue

                # Normalize and compute NCC
                i1 = (prev_crop - prev_crop.mean()) / (prev_crop.std() + 1e-8)
                i2 = (curr_crop - curr_crop.mean()) / (curr_crop.std() + 1e-8)
                ncc = float(np.mean(i1 * i2))
                tracks[tid]['sim_list'].append(ncc)
                tracks[tid]['time_list'].append(current_time / 1e6)

                # Update track bbox/centroid to current detection
                tracks[tid]['bbox'] = det_bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                tracks[tid]['centroid'] = (cx, cy)

                # Estimate RPM occasionally if we have enough samples
                sim_arr = np.array(tracks[tid]['sim_list'])
                if sim_arr.size >= 40:
                    try:
                        # estimate_rpm_wavelet returns (time_s, rpm_smooth) where rpm_smooth is freq*60
                        _, rpm_smooth = estimate_rpm_wavelet(np.array(tracks[tid]['time_list']), sim_arr, num_blades=2, plot=False)
                        finite = rpm_smooth[np.isfinite(rpm_smooth)]
                        if finite.size > 0:
                            # Divide by number of blades to get rotations per minute
                            rpm_est = float(np.nanmedian(finite)) / 2.0
                            tracks[tid]['rpm'] = rpm_est
                    except Exception:
                        pass

        else:
            # initialize sim_list/time_list for matched tracks even if no prev frame
            for tid in matched.keys():
                tracks[tid].setdefault('sim_list', [])
                tracks[tid].setdefault('time_list', [])

        prev_img = display_image.copy()

        # Remove stale tracks
        to_delete = []
        for tid, track in list(tracks.items()):
            if frame_idx - track.get('last_seen', frame_idx) > MAX_TRACK_LOST:
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]

        # Draw tracks and RPM labels
        vis = result_image.copy()
        for tid, track in tracks.items():
            x1, y1, x2, y2 = [int(v) for v in track['bbox']]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"ID {tid}"
            cv2.putText(vis, label, (x1, max(12, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (x1, max(12, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            rpm_val = track.get('rpm', float('nan'))
            if rpm_val is None or (isinstance(rpm_val, float) and np.isnan(rpm_val)):
                rpm_label = "RPM: --"
            else:
                rpm_label = f"RPM: {rpm_val:.1f}"
            cv2.putText(vis, rpm_label, (x1, max(28, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, rpm_label, (x1, max(28, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Display time and show
        cv2.putText(vis, f"t: {current_time/1000:.3f} ms", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Event Time Surface with Circles", vis)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            print(f"Playback {'PAUSED' if paused else 'RESUMED'}")

        frame_idx += 1

    # Cleanup
    cv2.destroyAllWindows()
    print("Visualization finished.")


# --- Execution ---
if __name__ == "__main__":
    print("This module provides `visualize_time_surface`. Run `main.py` to launch the visualizer or import functions into your script.")
