from ultralytics import YOLO

model = YOLO(r"/Users/kaungkhantlin/Developer/2_2025/AI_Concepts/project/best.pt")
results = model.predict(source=r"/Users/kaungkhantlin/Developer/2_2025/AI_Concepts/project/test/IMG_7757.JPG", conf=0.25, save=True)
print("Saved to:", results[0].save_dir)
