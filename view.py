import cv2
import argparse
import os

def load_detections(detections_file_path):
    """
    Load detections from a text file.
    Expected format for each line:
      frame_number num_detections x y w h conf [x y w h conf ...]
    Returns a dictionary mapping frame number (int) to a list of detections.
    Each detection is a tuple: (x, y, w, h, conf)
    """
    detections = {}
    with open(detections_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            frame_number = int(parts[0])
            num_dets = int(parts[1])
            det_list = []
            idx = 2
            for _ in range(num_dets):
                try:
                    x = int(float(parts[idx]))
                    y = int(float(parts[idx+1]))
                    w = int(float(parts[idx+2]))
                    h = int(float(parts[idx+3]))
                    conf = float(parts[idx+4])
                except (IndexError, ValueError):
                    break
                det_list.append((x, y, w, h, conf))
                idx += 5
            detections[frame_number] = det_list
    return detections

def main():
    parser = argparse.ArgumentParser(description="Display video with detections overlaid and optionally save the output.")
    parser.add_argument("video", help="Path to the video file.")
    parser.add_argument("detections", help="Path to the detections file.")
    parser.add_argument("--save", action="store_true", help="If set, saves the video with detections drawn.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print("Error: Video file not found:", args.video)
        return

    if not os.path.exists(args.detections):
        print("Error: Detections file not found:", args.detections)
        return

    # Load detections from file.
    det_dict = load_detections(args.detections)

    # Open the video.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Prepare video writer if saving is requested.
    writer = None
    if args.save:
        output_dir = "output_video"
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, "video_with_detections.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Saving video with detections to: {output_video_path}")

    frame_number = 1
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw detections if available for this frame.
        if frame_number in det_dict:
            for (x, y, w, h, conf) in det_dict[frame_number]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Detections", frame)
        if args.save and writer is not None:
            writer.write(frame)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        frame_number += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
