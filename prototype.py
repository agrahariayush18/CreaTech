import streamlit as st
import cv2
import numpy as np

# Function for Real-Time Monitoring
def real_time_monitoring():
    # Start video capture
    cap = cv2.VideoCapture(0)  # Use webcam for real-time video capture

    # Add a placeholder for the live video feed
    video_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video from webcam.")
            break

        # Convert the frame to grayscale and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Convert the OpenCV image (BGR) to RGB format for Streamlit
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Display the processed image in the Streamlit app
        video_placeholder.image(edges_rgb, channels="RGB", caption="Live Edge Detection")

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Streamlit App
st.title("Real-Time Project Monitoring Dashboard")
st.subheader("AI-Powered Live Video Feed with Edge Detection")

# Checkbox to start real-time monitoring
run_monitoring = st.checkbox("Start Real-Time Monitoring")
if run_monitoring:
    real_time_monitoring()