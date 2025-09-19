import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Car Detector", layout="wide")
st.title("Car Detector")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    cam_index = st.number_input("Camera index", min_value=0, value=0, step=1)
    conf = st.slider("Confidence", 0.1, 0.9, 0.5, 0.05)
    jpeg_quality = st.slider("JPEG quality (display)", 50, 100, 80, 1)
    run = st.toggle("Run detection", value=False, key="run_toggle")

# Load YOLO model once
if "model" not in st.session_state:
    st.session_state.model = YOLO("yolov8n.pt")
CAR_CLASS_ID = 2  # COCO: car

# Camera resource
if "cap" not in st.session_state:
    st.session_state.cap = None

# Unique counting state
if "total_cars" not in st.session_state:
    st.session_state.total_cars = 0
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()

with st.sidebar:
    if st.button("Reset counters"):
        st.session_state.total_cars = 0
        st.session_state.seen_ids = set()

def ensure_camera(idx: int):
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            st.error(f"Could not open camera index {idx}")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        st.session_state.cap = cap
    return st.session_state.cap

def release_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
    st.session_state.cap = None

# UI placeholders
meta_col, fps_col = st.columns([3, 1])
frame_count_text = meta_col.metric("Cars in frame", 0)
total_count_text = meta_col.metric("Total unique cars", 0)
fps_text = fps_col.metric("FPS", 0.0)
frame_placeholder = st.empty()

def draw_header(frame, text: str):
    h, w = frame.shape[:2]
    bar_h = 40
    cv2.rectangle(frame, (0, 0), (w, bar_h), (32, 32, 32), -1)
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

if run:
    cap = ensure_camera(cam_index)
    if cap:
        prev = time.time()
        fps = 0.0
        try:
            while st.session_state.get("run_toggle", False):
                ok, frame = cap.read()
                if not ok:
                    st.warning("Could not read frame.")
                    break

                # ---- TRACK (ByteTrack) + filter to cars ----
                results = st.session_state.model.track(
                    frame,
                    conf=conf,
                    classes=[CAR_CLASS_ID],
                    persist=True,                 # keep tracks between frames
                    verbose=False,
                    tracker="bytetrack.yaml"      # built-in tracker config
                )[0]

                boxes = results.boxes
                ids = boxes.id if boxes is not None else None
                cars_in_frame = 0

                if boxes is not None and ids is not None:
                    for i in range(len(boxes)):
                        track_id = int(ids[i]) if ids[i] is not None else None
                        if track_id is None:
                            continue

                        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                        conf_val = float(boxes.conf[i])

                        # draw
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(
                            frame,
                            f"Car ID {track_id}  {conf_val:.2f}",
                            (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA
                        )

                        cars_in_frame += 1

                        # ---- unique counting: only count first time we see this ID ----
                        if track_id not in st.session_state.seen_ids:
                            st.session_state.seen_ids.add(track_id)
                            st.session_state.total_cars += 1

                # FPS
                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # Header + metrics
                draw_header(
                    frame,
                    f"Unique cars: {st.session_state.total_cars} | In frame: {cars_in_frame} | FPS: {fps:.1f}"
                )
                frame_count_text.metric("Cars in frame", cars_in_frame)
                total_count_text.metric("Total unique cars", st.session_state.total_cars)
                fps_text.metric("FPS", round(fps, 1))

                # Display
                ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok_jpg:
                    continue
                rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

        finally:
            release_camera()
else:
    release_camera()
    frame_placeholder.empty()
