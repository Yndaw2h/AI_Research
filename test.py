import cv2
from ultralytics import YOLO

def load_model(model_path):
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

def detect_objects(model, frame):
    results = model(frame)
    return results

def draw_boxes(frame, results):
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.names

    for i in range(len(boxes)):
        if scores[i] > 0.5:
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            label = labels[int(result.boxes.cls[i])]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(frame, f"{label} {score:.2f}", 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    model_path = "obj.pt"
    model = load_model(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(model, frame)

        frame_with_boxes = draw_boxes(frame, results)

        cv2.imshow("Object Detection", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()