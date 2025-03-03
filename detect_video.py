import cv2
import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="YOLO detection on a video using merged consecutive frames."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--save", action="store_true", help="If set, saves the detections to a file.")
    parser.add_argument("--weights", default="model_weights/best.pt", help="Path to the YOLO model weights.")
    args = parser.parse_args()

    # Load the YOLO model
    model = YOLO(args.weights)

    # Setup output file if saving detections
    if args.save:
        output_dir = "output_detection_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        detections_file_path = os.path.join(output_dir, "video_detections.txt")
        detections_file = open(detections_file_path, "w")
        print(f"Saving detection results to: {detections_file_path}")

    # Open the video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read the first two frames to initialize the sliding window
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    ret, second_frame = cap.read()
    # If the video has only one frame, duplicate it
    if not ret:
        second_frame = first_frame.copy()

    # Initialize sliding window: previous, current, next
    prev_frame = first_frame
    current_frame = first_frame
    next_frame = second_frame

    frame_counter = 0

    cv2.namedWindow('Detection Evaluation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection Evaluation', 1080, 640)

    while True:
        frame_counter += 1

        # Create merged frame using previous, current, and next frames
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        merged_frame = cv2.merge([prev_gray, current_gray, next_gray])

        # Run YOLO detection on the merged frame
        results = model(merged_frame, imgsz=1280, conf=0.15, iou=0.5)
        detections = results[0]
        detected_boxes = []  # Each entry: [x, y, w, h, conf]
        boxes = detections.boxes.data.cpu().numpy()
        for det in boxes:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.1:  # Confidence threshold
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                detected_boxes.append([x, y, w, h, conf])

        # Draw detection boxes on the current frame (green rectangles)
        for det in detected_boxes:
            x, y, w, h, conf = det
            cv2.rectangle(current_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(current_frame, f"{conf:.2f}", (int(x), int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Detection Evaluation', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

        # Optionally, write the detections for the current frame to the file
        if args.save:
            # Format: frame_number num_detections [x y w h conf]...
            line_parts = [str(frame_counter), str(len(detected_boxes))]
            for det in detected_boxes:
                x, y, w, h, conf = det
                line_parts.extend([str(int(x)), str(int(y)), str(int(w)), str(int(h)), f"{conf:.2f}"])
            detections_file.write(" ".join(line_parts) + "\n")

        # Update sliding window: shift frames forward
        prev_frame = current_frame
        current_frame = next_frame
        ret, next_frame = cap.read()
        if not ret:
            # If no more frames are available, use the current frame as the next frame
            next_frame = current_frame.copy()
            # Break out if we've reached the end of the video
            break

    cap.release()
    if args.save:
        detections_file.close()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()
