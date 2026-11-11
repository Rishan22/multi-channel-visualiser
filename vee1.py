import cv2

def apply_heatmap(image):
   
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap

def process_and_display_heatmaps(frame):
    # Convert to HSV and LAB
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Extract channels
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    l_channel = lab[:, :, 0]

    # Normalize channels 
    h_channel = cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX)
    s_channel = cv2.normalize(s_channel, None, 0, 255, cv2.NORM_MINMAX)
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)

    # Operations
    broad_map = cv2.addWeighted(h_channel, 0.5, s_channel, 0.5, 0)  # (H + S)
    combined_map = cv2.bitwise_or(broad_map, l_channel)  # (H + S) | L
    shadows_removed = cv2.subtract(l_channel, s_channel)  # L - S
    highlights_removed = cv2.subtract(l_channel, broad_map)  # L - (H + S)

    # Apply heatmaps
    heatmaps = [
        ("Original L Channel", apply_heatmap(l_channel)),
        ("Broad Environment Map (H + S)", apply_heatmap(broad_map)),
        ("Combined Map (H + S) | L", apply_heatmap(combined_map)),
        ("Shadows Removed (L - S)", apply_heatmap(shadows_removed)),
        ("Highlights Removed (L - (H + S))", apply_heatmap(highlights_removed)),
    ]

    # Arrange heatmaps in a grid (two rows)
    row1 = cv2.hconcat([heatmaps[0][1], heatmaps[1][1], heatmaps[2][1]])
    row2 = cv2.hconcat([heatmaps[3][1], heatmaps[4][1], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)])

   
    grid_display = cv2.vconcat([row1, row2])

    # Display grid
    cv2.imshow("Multi-Panel Heatmap Visualization", grid_display)

# Webcam capture setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the current frame and display heatmaps
        process_and_display_heatmaps(frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
