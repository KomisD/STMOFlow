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
    parser.add_argument(
        "--colab",
        action="store_true",
        help="If set, writes annotated video out and skips GUI display (for Colab)."
    )
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
    if not ret:
        second_frame = first_frame.copy()

    # Initialize sliding window frames
    prev_frame = first_frame
    current_frame = first_frame
    next_frame = second_frame
    frame_counter = 0

    # Setup writer for Colab (headless) mode
    if args.colab:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs("output_detection_evaluation", exist_ok=True)
        out_path = os.path.join("output_detection_evaluation", "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    else:
        # GUI mode: create window
        cv2.namedWindow('Detection Evaluation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection Evaluation', 1080, 640)

    # Main processing loop
    while True:
        frame_counter += 1

        # Merge previous, current, next into a 3-channel input
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        merged_frame = cv2.merge([prev_gray, current_gray, next_gray])

        # Run inference
        results = model(merged_frame, imgsz=1280, conf=0.15, iou=0.5)
        detections = results[0]
        boxes = detections.boxes.data.cpu().numpy()

        # Collect high-confidence detections
        detected_boxes = []
        for det in boxes:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.1:
                w = x2 - x1
                h = y2 - y1
                detected_boxes.append([x1, y1, w, h, conf])

        # Draw boxes on current_frame
        for x, y, w, h, conf in detected_boxes:
            cv2.rectangle(current_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(
                current_frame,
                f"{conf:.2f}",
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Output frame: write or show
        if args.colab:
            writer.write(current_frame)
        else:
            cv2.imshow('Detection Evaluation', current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user.")
                break

        # Save detections to file if requested
        if args.save:
            parts = [str(frame_counter), str(len(detected_boxes))]
            for x, y, w, h, conf in detected_boxes:
                parts.extend([str(int(x)), str(int(y)), str(int(w)), str(int(h)), f"{conf:.2f}"])
            detections_file.write(" ".join(parts) + "\n")

        # Shift sliding window
        prev_frame = current_frame
        current_frame = next_frame
        ret, next_frame = cap.read()
        if not ret:
            next_frame = current_frame.copy()
            break

    # Release resources
    cap.release()
    if args.colab:
        writer.release()
        print(f"Annotated video saved to: {out_path}")
    if args.save:
        detections_file.close()
    if not args.colab:
        cv2.destroyAllWindows()

    print("Processing complete.")


if __name__ == "__main__":
    main()
