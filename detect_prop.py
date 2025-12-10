import argparse
import numpy as np
import cv2
import os
from metavision_core.event_io import EventsIterator

# --- Configuration ---
# Default resolution for a common event camera (Prophesee VGA)
# Adjust if your data has a different resolution
WIDTH, HEIGHT = 1280, 720

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Visualize events from a .raw file using Time Surface visualization.")
parser.add_argument("--input_raw", type=str, help="Path to input .raw file")
parser.add_argument("--delta_t", type=int, default=10000, help="Delta time for EventsIterator (in us, default: 10000)")
args = parser.parse_args()

# --- Main variables ---
raw_path = os.path.expanduser(args.input_raw)
delta_t = args.delta_t

# Check if file exists
if not os.path.exists(raw_path):
    print(f"Error: Input file not found at {raw_path}")
    exit(1)

# Initialize the Metavision EventsIterator
try:
    mv_iterator = EventsIterator(input_path=raw_path, delta_t=delta_t)
except Exception as e:
    print(f"Error initializing EventsIterator: {e}")
    exit(1)

# Get sensor size from iterator metadata (if available)
try:
    if mv_iterator.get_size() is not None:
        HEIGHT, WIDTH = mv_iterator.get_size()
except:
    # Fallback to default if metadata retrieval fails
    print(f"Warning: Could not retrieve sensor size. Using default: {WIDTH}x{HEIGHT}")

# --- Circle Detection Function ---
def detect_circles(image, min_radius=30, max_radius=100, param1=50, param2=30):
    """
    Detects circles in an image, with denoising for salt-and-pepper noise.
    
    Parameters:
        image (ndarray): The input image.
        min_radius (int): The minimum radius of the circles to detect.
        max_radius (int): The maximum radius of the circles to detect.
        param1 (int): The higher threshold for edge detection.
        param2 (int): The threshold for center detection.

    Returns:
        result_image (ndarray): The image with detected circles drawn on it.
        circles (ndarray or None): Array of detected circles (None if no circles detected).
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median filtering to reduce salt-and-pepper noise
    gray_denoised = cv2.medianBlur(gray, 5)  # Kernel size can be adjusted
    
    # Apply GaussianBlur to smooth the image further (optional)
    gray_blurred = cv2.GaussianBlur(gray_denoised, (15, 15), 0)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,  # Detection method
        dp=1,  # The inverse ratio of resolution
        minDist=30,  # Minimum distance between detected centers
        param1=param1,  # Higher threshold for edge detection
        param2=param2,  # Threshold for center detection
        minRadius=min_radius,  # Minimum circle radius
        maxRadius=max_radius  # Maximum circle radius
    )
    
    # Draw circles if detected
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Convert to integer values
        for i in circles[0, :]:
            center = (i[0], i[1])  # The center of the circle
            radius = i[2]  # The radius of the circle
            
            # # Draw the circle's center
            # cv2.circle(image, center, 2, (0, 255, 0), 3)
            
            # Draw the circle itself
            cv2.circle(image, center, radius, (0, 0, 255), 3)
    
    return image, circles  # Return the image with circles drawn, and the circle data


# --- Visualization Function ---
def visualize_time_surface(mv_iterator, width, height):
    """
    Iterates through events and visualizes them as a Time Surface.
    
    The Time Surface is an image where the pixel intensity represents 
    the time difference since the last event at that pixel location.
    
    Newer events (more recent) are brighter; older events fade away.
    """
    print(f"Starting visualization for {width}x{height} resolution.")
    print("Press 'SPACE' to pause/resume. Press 'q' or 'ESC' to quit.")

    # Time Surface (TS) buffer: stores the timestamp of the last event at each pixel
    # Initialized to a very old time (0)
    ts_surface = np.zeros((height, width), dtype=np.uint64)
    
    # Image buffer for visualization (will be converted to BGR for display)
    # Intensity will be based on time difference
    display_image = np.zeros((height, width), dtype=np.uint8)
    
    # Maximum time difference (in us) to be considered "bright"
    # Events older than this threshold will be completely black/faded.
    FADE_TIME = 10000  # 10 ms (10,000 us)

    # Variables for controlling playback
    paused = False
    
    # Process events chunk by chunk
    for evts in mv_iterator:
        if evts.size == 0:
            # Check for key press even if no events were processed in this chunk
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27: # 'q' or ESC
                break
            elif key == ord(' '): # SPACE
                paused = not paused
            
            # If paused, check for commands until unpaused
            while paused:
                key = cv2.waitKey(1)
                if key == ord(' '): # SPACE
                    paused = not paused
                elif key == ord('q') or key == 27:
                    return # Exit the function
            
            continue # Skip to next chunk if no events

        # --- Update Time Surface ---
        
        # Get event data
        x = evts['x']
        y = evts['y']
        t = evts['t']
        
        # Update the TS buffer with the latest timestamp for each pixel that fired
        ts_surface[y, x] = t

        # --- Generate Time Surface Image ---

        # Get the current maximum time
        current_time = t[-1] 
        
        # Calculate the age of the last event for every pixel
        time_diff = current_time - ts_surface
        
        # Create the intensity map (0-255)
        
        # Clip the age to the maximum fade time
        clipped_diff = np.clip(time_diff, 0, FADE_TIME)
        
        # Normalize and scale to 0-255 (invert: 255 for age=0, 0 for age>=FADE_TIME)
        intensity = 255 * (1.0 - clipped_diff / FADE_TIME)
        
        # Convert to 8-bit image for display
        display_image = intensity.astype(np.uint8)
        
        # Convert grayscale (Time Surface) to BGR for consistent display
        display_bgr = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        # --- Detect Circles ---
        result_image, detected_circles = detect_circles(display_bgr)

        # --- Display the result ---
        # Display current time in the image (optional)
        cv2.putText(result_image, f"t: {current_time/1000:.3f} ms", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Show the time surface with detected circles
        cv2.imshow("Event Time Surface with Circles", result_image)
        
        # Handle user input (waiting for 1 millisecond)
        key = cv2.waitKey(1) 
        
        if key == ord('q') or key == 27: # 'q' or ESC
            break
        elif key == ord(' '): # SPACE
            paused = not paused
            print(f"Playback {'PAUSED' if paused else 'RESUMED'}")
        
        # If paused, wait indefinitely until a resume or quit command is given
        while paused:
            key = cv2.waitKey(100) # Wait longer while paused
            if key == ord(' '):
                paused = not paused
                print("Playback RESUMED")
                break
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return

    # Cleanup
    cv2.destroyAllWindows()
    print("Visualization finished.")


# --- Execution ---
if __name__ == "__main__":
    visualize_time_surface(mv_iterator, WIDTH, HEIGHT)
