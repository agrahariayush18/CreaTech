import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO  # For object detection

# Load Pre-trained YOLO Model
object_detection_model = YOLO("yolov8n.pt")  # YOLO model for object detection

# Function for Object Detection and Worker Activity Analysis
def detect_objects_and_analyze_workers(frame, previous_worker_positions):
    results = object_detection_model(frame)
    detected_objects = []
    current_worker_positions = {}

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = object_detection_model.names[cls_id]
            confidence = float(box.conf[0])

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            detected_objects.append(class_name)

            # Analyze worker activity
            if class_name == "person":
                worker_id = len(current_worker_positions)  # Assign a unique ID to each worker
                current_worker_positions[worker_id] = (x1, y1, x2, y2)

                # Compare with previous positions to determine activity
                if worker_id in previous_worker_positions:
                    prev_x1, prev_y1, prev_x2, prev_y2 = previous_worker_positions[worker_id]
                    movement_threshold = 10  # Define a threshold for movement
                    if abs(x1 - prev_x1) > movement_threshold or abs(y1 - prev_y1) > movement_threshold:
                        worker_status = "Active"
                    else:
                        worker_status = "Idle"
                else:
                    worker_status = "Idle"

                # Display worker status
                cv2.putText(frame, worker_status, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, detected_objects, current_worker_positions

# Function for Real-Time Monitoring
def real_time_monitoring():
    cap = cv2.VideoCapture(0)  # Use webcam for real-time video capture
    video_placeholder = st.empty()
    previous_worker_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video from webcam.")
            break

        # Step 1: Object Detection and Worker Activity Analysis
        frame, detected_objects, current_worker_positions = detect_objects_and_analyze_workers(frame, previous_worker_positions)

        # Update previous worker positions
        previous_worker_positions = current_worker_positions

        # Convert the OpenCV image (BGR) to RGB format for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed image in the Streamlit app
        video_placeholder.image(frame_rgb, channels="RGB", caption="Live Monitoring")

        # Display Detected Objects and Worker Status
        st.sidebar.subheader("Detected Objects")
        st.sidebar.write(detected_objects)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit App
st.title("Real-Time Construction Site Monitoring Dashboard")
st.subheader("AI-Powered Live Video Feed with Object Detection and Worker Activity Analysis")

# Sidebar for Project Details
st.sidebar.title("Project Overview")
st.sidebar.markdown("""
- **Problem Statement**: Delays, cost overruns, and lack of real-time monitoring in Indian construction projects.
- **Solution**: AI-powered system using computer vision for progress tracking, resource optimization, and safety monitoring.
- **Impact**: Up to 20% reduction in delays, 15% cost savings, improved safety.
""")

# Checkbox to start real-time monitoring
run_monitoring = st.checkbox("Start Real-Time Monitoring")
if run_monitoring:
    real_time_monitoring()