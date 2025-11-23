import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class ID for person


def density_label(count, low_limit, high_limit):
    if count < low_limit:
        return "LOW", (60, 180, 60)
    if count < high_limit:
        return "MEDIUM", (0, 200, 255)
    return "HIGH", (0, 0, 255)


def annotate(frame, detections, low_limit, high_limit, alert_limit):
    count = len(detections)
    label, color = density_label(count, low_limit, high_limit)

    for x1, y1, x2, y2, confidence in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.putText(frame, f"Person {confidence:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 112), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.putText(frame, f"People detected: {count}", (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Crowd level: {label}", (22, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    message = "ALERT: CROWD LIMIT REACHED" if count >= alert_limit else "Press Q to quit"
    cv2.putText(frame, message, (22, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 0, 255) if count >= alert_limit else (230, 230, 230), 2)
    return frame, count, label


def main():
    parser = argparse.ArgumentParser(description="Crowd estimation demo using YOLO person detection.")
    parser.add_argument("--source", default="0", help="Webcam index (0) or path to a video file.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights path.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence from 0 to 1.")
    parser.add_argument("--low-limit", type=int, default=5, help="Below this count is LOW.")
    parser.add_argument("--high-limit", type=int, default=15, help="Below this count (and >= low) is MEDIUM.")
    parser.add_argument("--alert-limit", type=int, default=20, help="Alert when count reaches this value.")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video to outputs/.")
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.low_limit < 0 or args.high_limit <= args.low_limit:
        parser.error("Require 0 <= low-limit < high-limit")
    if args.alert_limit < 1:
        parser.error("--alert-limit must be at least 1")

    print("Loading YOLO model. First run may download pretrained weights.")
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}. Check video path or camera permissions.")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    log_path = output_dir / "crowd_log.csv"
    writer = None

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        log = csv.writer(log_file)
        log.writerow(["timestamp", "people_detected", "crowd_level", "alert"])
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = model.predict(frame, classes=[PERSON_CLASS_ID],
                                       conf=args.confidence, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        confidence = float(box.conf[0].cpu().item())
                        detections.append((x1, y1, x2, y2, confidence))

                annotated, count, level = annotate(
                    frame, detections, args.low_limit, args.high_limit, args.alert_limit
                )
                alert = count >= args.alert_limit
                log.writerow([datetime.now().isoformat(timespec="seconds"), count, level, int(alert)])

                if args.save_video and writer is None:
                    h, w = annotated.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0 or fps > 120:
                        fps = 20.0
                    writer = cv2.VideoWriter(
                        str(output_dir / "annotated_crowd.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                if writer is not None:
                    writer.write(annotated)

                cv2.imshow("Crowd Density Estimation", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    print(f"Saved crowd log to {log_path}")


if __name__ == "__main__":
    main()
