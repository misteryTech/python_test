from ultralytics import YOLO
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"C:\xampp\htdocs\python_test\bottle\YOLODataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device="0",      # ← GPU
        workers=0,
        name="bottle_qc_final"
    )

    print("Done! Model saved in runs/detect/bottle_qc_final/weights/best.pt")