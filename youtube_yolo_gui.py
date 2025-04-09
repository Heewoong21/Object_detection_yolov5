import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import torch
from PIL import Image, ImageTk
from yt_dlp import YoutubeDL

# 디바이스 설정: GPU 사용 가능하면 'cuda', 아니면 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# YOLOv5 모델 로드 및 GPU로 이동
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# 유튜브 스트리밍 URL 가져오기 함수
def get_stream_url(youtube_url, quality_option):
    # yt-dlp 포맷 맵
    format_map = {
        "최고화질": "best[ext=mp4]",
        "보통화질": "best[height<=480][ext=mp4]",
        "낮은화질": "worst[ext=mp4]"
    }

    ydl_opts = {
        'format': format_map.get(quality_option, "best[ext=mp4]"),
        'quiet': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# 스트리밍 시작
def start_detection():
    youtube_url = url_entry.get()
    quality_option = quality_var.get()

    if not youtube_url:
        messagebox.showerror("Error", "유튜브 URL을 입력하세요.")
        return

    try:
        stream_url = get_stream_url(youtube_url, quality_option)
        cap = cv2.VideoCapture(stream_url)
    except Exception as e:
        messagebox.showerror("Error", f"유튜브 영상을 열 수 없습니다: {e}")
        return

    if not cap.isOpened():
        messagebox.showerror("Error", "유튜브 스트리밍을 열 수 없습니다.")
        return

    def process_frame():
        ret, frame = cap.read()
        if not ret:
            messagebox.showinfo("Info", "스트리밍이 종료되었습니다.")
            cap.release()
            return

        # YOLOv5 추론
        results = model(frame)  # YOLOv5가 자동으로 GPU로 처리
        detections = results.xyxy[0].cpu().numpy()

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, process_frame)

    process_frame()

# GUI 생성
root = tk.Tk()
root.title("YOLOv5 YouTube Stream Object Detection")

# YouTube URL 입력
tk.Label(root, text="YouTube URL:").pack(pady=5)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

# 화질 선택 드롭다운
tk.Label(root, text="화질 선택:").pack(pady=5)
quality_var = tk.StringVar()
quality_dropdown = ttk.Combobox(root, textvariable=quality_var, state="readonly")
quality_dropdown['values'] = ("최고화질", "보통화질", "낮은화질")
quality_dropdown.current(0)
quality_dropdown.pack(pady=5)

# 시작 버튼
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

# 비디오 프레임 출력
video_label = tk.Label(root)
video_label.pack()

# GUI 실행
root.mainloop()
