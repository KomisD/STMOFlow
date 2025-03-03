import cv2
import os
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_tracker():
    return DeepSort(
        max_age=10,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.7,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True
    )

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    Boxes are in [x, y, w, h] format.
    """
    b1_x1, b1_y1, b1_w, b1_h = box1
    b2_x1, b2_y1, b2_w, b2_h = box2

    b1_x2 = b1_x1 + b1_w
    b1_y2 = b1_y1 + b1_h
    b2_x2 = b2_x1 + b2_w
    b2_y2 = b2_y1 + b2_h

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = b1_w * b1_h
    area2 = b2_w * b2_h
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection and DeepSORT tracking on a video file. "
                    "Green: Detections, Red: Tracker boxes, Blue: Final boxes."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--save", action="store_true", help="Save final detection boxes (blue) to a file.")
    parser.add_argument("--weights", default="model_weights/best.pt", help="Path to the YOLO model weights.")
    args = parser.parse_args()

    # Load YOLO model.
    model = YOLO(args.weights)

    # Initialize DeepSORT tracker.
    tracker = initialize_tracker()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Prepare file for saving final boxes if --save is enabled.
    if args.save:
        output_dir = "output_tracking_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        detections_file_path = os.path.join(output_dir, "video_final_detections.txt")
        detections_file = open(detections_file_path, "w")
        print(f"Saving final detections to: {detections_file_path}")

    cv2.namedWindow('Detection, Tracking, and Final', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection, Tracking, and Final', 1080, 640)

    # Read the first three frames for a sliding window.
    ret, frame_prev = cap.read()
    if not ret:
        print("Error: Unable to read first frame.")
        return
    ret, frame_curr = cap.read()
    if not ret:
        frame_curr = frame_prev.copy()
    ret, frame_next = cap.read()
    if not ret:
        frame_next = frame_curr.copy()

    frame_counter = 0

    while True:
        frame_counter += 1

        # Create merged frame from previous, current, and next frames.
        prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
        merged_frame = cv2.merge([prev_gray, curr_gray, next_gray])

        # Run YOLO detection on the merged frame.
        results = model(merged_frame, imgsz=1280, conf=0.5, iou=0.15)
        detections = results[0]
        detected_boxes_with_conf = []  # Detections for visualization (green)
        detection_list = []  # For tracker update: ([x,y,w,h], conf, 'object')
        boxes = detections.boxes.data.cpu().numpy()
        for det in boxes:
            x1, y1, x2, y2, conf, cls = det
            if conf > 0.1:
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1
                detected_boxes_with_conf.append([x, y, w, h, conf])
                detection_list.append(([x, y, w, h], conf, 'object'))

        # Update tracker with current detections.
        tracks = tracker.update_tracks(detection_list, frame=frame_curr)
        tracked_boxes = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            tracked_boxes.append([x1, y1, x2 - x1, y2 - y1])

        # Determine final boxes (blue) using the following logic:
        #   1. If both detection and tracker boxes exist, filter detections by IoU > 0.1 with any tracker box.
        #      If none pass, fall back to tracker boxes (with default confidence 1.0).
        #   2. If only detections exist, use them.
        #   3. If only tracker boxes exist, use them.
        if detected_boxes_with_conf and tracked_boxes:
            filtered_detections = []
            for detection in detected_boxes_with_conf:
                for trk in tracked_boxes:
                    if calculate_iou(detection[:4], trk) > 0.1:
                        filtered_detections.append(detection)
                        break
            if filtered_detections:
                final_boxes = filtered_detections
            else:
                final_boxes = [[x, y, w, h, 1.0] for (x, y, w, h) in tracked_boxes]
        elif detected_boxes_with_conf:
            final_boxes = detected_boxes_with_conf
        elif tracked_boxes:
            final_boxes = [[x, y, w, h, 1.0] for (x, y, w, h) in tracked_boxes]
        else:
            final_boxes = []

        # Visualization:
        vis_frame = frame_curr.copy()
        # Draw YOLO detections in green.
        for det in detected_boxes_with_conf:
            x, y, w, h, conf = det
            cv2.rectangle(vis_frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        # Draw tracker boxes in red.
        for trk in tracked_boxes:
            x, y, w, h = trk
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Draw final boxes in blue on top.
        for final in final_boxes:
            x, y, w, h, conf = final
            cv2.rectangle(vis_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)



        cv2.imshow('Detection, Tracking, and Final', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # If save is enabled, write only the final (blue) boxes.
        if args.save:
            # Format: frame_number num_final_boxes x y w h conf (for each box)
            line_parts = [str(frame_counter), str(len(final_boxes))]
            for box in final_boxes:
                x, y, w, h, conf = box
                line_parts.extend([str(int(x)), str(int(y)), str(int(w)), str(int(h)), f"{conf:.2f}"])
            detections_file.write(" ".join(line_parts) + "\n")

            # Update sliding window: shift frames.
        frame_prev = frame_curr
        frame_curr = frame_next
        ret, frame_next = cap.read()
        if not ret:
            break  # Exit the loop when the video ends.
        # Duplicate last frame if video ends.

    cap.release()
    if args.save:
        detections_file.close()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()
