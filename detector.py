import os
from ultralytics import YOLO
import cv2

MODEL_PATH = os.path.join('models', 'best.pt')
model = YOLO(MODEL_PATH)

STAGE_RECOMMENDATIONS = {
    '0': 'No symptoms. Monitor regularly.',
    '1': 'Initial spots. Consider natural remedies.',
    '3': 'Mild infection. Apply fungicide A.',
    '5': 'Moderate infection. Use fungicide B.',
    '7': 'Severe infection. Isolate affected area.',
    '9': 'Critical. Consult an agronomist immediately.'
}

def detect_stage(image_path, output_dir="static/outputs"):
    results = model(image_path)[0]
    img = cv2.imread(image_path)
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf*100:.1f}%)"
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        detections.append({
            "stage": label,
            "confidence": f"{conf*100:.2f}%",
            "recommendation": STAGE_RECOMMENDATIONS.get(label, "No recommendation available")
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"detection_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img)
    return output_path.replace("\\", "/"), detections
