import streamlit as st
import cv2
import numpy as np

# Load Pre-trained MobileNet SSD Model
prototxt_path = "/Users/ayushagrahari/Desktop/CreaTech/deploy (4).prototxt"
model_path = "/Users/ayushagrahari/Desktop/CreaTech/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# List of class labels for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Function for Object Detection
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter out weak detections
            class_id = int(detections[0, 0, i, 1])
            class_name = CLASSES[class_id]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Streamlit App
st.title("Real-Time Object Detection with MobileNet SSD")
run = st.checkbox("Start Camera")
video_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        break

    frame = detect_objects(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame, channels="RGB")

cap.release()