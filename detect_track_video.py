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
        description="Run YOLO detection and DeepSORT tracking on a video file."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--save", action="store_true",
                        help="Save final detection boxes (blue) to a file.")
    parser.add_argument("--weights", default="model_weights/best.pt",
                        help="Path to the YOLO model weights.")
    parser.add_argument("--colab", action="store_true",
                        help="If set, writes annotated video out and skips GUI display (for Colab).")
    args = parser.parse_args()

    # Load YOLO model and tracker
    model = YOLO(args.weights)
    tracker = initialize_tracker()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Prepare save file
    if args.save:
        out_dir = "output_tracking_evaluation"
        os.makedirs(out_dir, exist_ok=True)
        det_path = os.path.join(out_dir, "video_final_detections.txt")
        det_file = open(det_path, "w")
        print(f"Saving final detections to: {det_path}")

    # Colab video writer setup
    if args.colab:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs("output_tracking_evaluation", exist_ok=True)
        out_vid = os.path.join("output_tracking_evaluation", "annotated_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("Error: could not open VideoWriter.")
            return
    else:
        cv2.namedWindow('Detect/Track/Final', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detect/Track/Final', 1080, 640)

    # Read initial sliding window frames
    ret, prev = cap.read()
    if not ret:
        print("Error: Unable to read first frame.")
        return
    ret, curr = cap.read()
    if not ret:
        curr = prev.copy()
    ret, nxt = cap.read()
    if not ret:
        nxt = curr.copy()

    frame_idx = 0
    while True:
        frame_idx += 1
        # merge frames
        pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        cg = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        ng = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
        merged = cv2.merge([pg, cg, ng])

        # detection
        res = model(merged, imgsz=1280, conf=0.5, iou=0.15)[0]
        boxes_np = res.boxes.data.cpu().numpy()
        det_vis = []
        det_list = []
        for x1,y1,x2,y2,conf,cls in boxes_np:
            if conf > 0.1:
                w_box, h_box = x2-x1, y2-y1
                det_vis.append([x1,y1,w_box,h_box,conf])
                det_list.append(([x1,y1,w_box,h_box], conf, 'object'))

        # tracking
        tracks = tracker.update_tracks(det_list, frame=curr)
        track_boxes = []
        for t in tracks:
            if not t.is_confirmed(): continue
            ltrb = t.to_ltrb()
            x1,y1,x2,y2 = map(int, ltrb)
            track_boxes.append([x1,y1,x2-x1,y2-y1])

        # final boxes
        if det_vis and track_boxes:
            filt = [d for d in det_vis if any(calculate_iou(d[:4], tb)>0.1 for tb in track_boxes)]
            final = filt if filt else [[*tb,1.0] for tb in track_boxes]
        elif det_vis:
            final = det_vis
        elif track_boxes:
            final = [[*tb,1.0] for tb in track_boxes]
        else:
            final = []

        # visualize
        vis = curr.copy()
        for x,y,w_box,h_box,conf in det_vis:
            cv2.rectangle(vis, (int(x),int(y)), (int(x+w_box),int(y+h_box)), (0,255,0),2)
        for x,y,w_box,h_box in track_boxes:
            cv2.rectangle(vis, (x,y), (x+w_box,y+h_box), (0,0,255),2)
        for x,y,w_box,h_box,conf in final:
            cv2.rectangle(vis, (int(x),int(y)), (int(x+w_box),int(y+h_box)), (255,0,0),2)

        # output
        if args.colab:
            writer.write(vis)
        else:
            cv2.imshow('Detect/Track/Final', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # save detections file
        if args.save:
            parts = [str(frame_idx), str(len(final))]
            for x,y,w_box,h_box,conf in final:
                parts.extend([str(int(x)),str(int(y)),str(int(w_box)),str(int(h_box)),f"{conf:.2f}"])
            det_file.write(" ".join(parts)+"\n")

        # shift
        prev, curr = curr, nxt
        ret, nxt = cap.read()
        if not ret:
            break

    cap.release()
    if args.colab:
        writer.release()
        print(f"Annotated tracking video saved to: {out_vid}")
    if args.save:
        det_file.close()
    if not args.colab:
        cv2.destroyAllWindows()
    print("Processing complete.")


if __name__ == "__main__":
    main()
