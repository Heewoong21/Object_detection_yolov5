import tkinter as tk
from tkinter import messagebox
import cv2
import torch
from PIL import Image, ImageTk

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 웹캠 스트리밍 함수
def start_detection():
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():
        messagebox.showerror("Error", "웹캠을 열 수 없습니다.")
        return

    def process_frame():
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "웹캠에서 프레임을 읽을 수 없습니다.")
            return

        # YOLOv5 모델로 추론
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # 바운딩 박스, 클래스, confidence

        # 바운딩 박스 그리기
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # OpenCV 이미지를 Tkinter에서 표시할 수 있도록 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, process_frame)

    process_frame()

# GUI 생성
root = tk.Tk()
root.title("YOLOv5 Object Detection")

# 시작 버튼
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack()

# 비디오 출력 라벨
video_label = tk.Label(root)
video_label.pack()

# GUI 실행
root.mainloop()