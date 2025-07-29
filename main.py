import cv2
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pywhatkit as kit
from joblib import load  # Import the load function


# Email alert function
def send_alert(subject, body):
    email_user = 'shobikadevi12@gmail.com'
    email_password = 'djzf avqt eolv dcep'  # Use environment variables for security
    email_send = 'shobikadevi41@gmail.com'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email_user, email_password)
            server.sendmail(email_user, email_send, msg.as_string())
            print("Email alert sent successfully!")
    except Exception as e:
        print(f"Failed to send email alert: {e}")

#load the model
loaded_model = load(r"trained_isolation_forest_model.pkl")
print("Model loaded successfully")

# WhatsApp alert function
def send_whatsapp_alert(message):
    try:
        # Get the current time and schedule the message 1 minute ahead
        current_hour = time.localtime().tm_hour
        current_minute = time.localtime().tm_min + 1  # Next minute
        phone_number = "+916374188937"  # Replace with the recipient's number

        # Send the WhatsApp message
        kit.sendwhatmsg(phone_number, message, current_hour, current_minute)
        print("WhatsApp alert sent successfully!")
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {e}")

# Load YOLO model
yolo_net = cv2.dnn.readNet(
    r'C:\Users\shobi\OneDrive\Desktop\rail\yolov3.weights',
    r'C:\Users\shobi\OneDrive\Desktop\rail\yolov3.cfg.txt'
)
with open(r'C:\Users\shobi\OneDrive\Desktop\rail\coco.names') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video source")

# Parameters
CROWD_THRESHOLD = 5
UNATTENDED_THRESHOLD = 60
FRAME_AVERAGE_COUNT = 10

# Tracking variables
bag_trackers = {}
bag_last_seen_time = {}
frame_counts = []

# Function to detect objects
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    height, width, _ = frame.shape
    boxes = []
    class_ids = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Reduced confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, class_ids, confidences

# Real-time video analysis
print("Starting real-time monitoring...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    boxes, class_ids, confidences = detect_objects(frame)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)  # Adjusted thresholds

    if indices is not None and len(indices) > 0:
        indices = indices.flatten()  # Flatten indices
        people_count = 0

        for i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            if class_ids[i] == 0:  # 'person' class
                people_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elif class_ids[i] == 67:  # 'backpack' class
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Bag: {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                bag_trackers[i] = (x, y, w, h)
                bag_last_seen_time[i] = time.time()

        # Update rolling average for people count
        if len(frame_counts) >= FRAME_AVERAGE_COUNT:
            frame_counts.pop(0)
        frame_counts.append(people_count)
        avg_people_count = sum(frame_counts) // len(frame_counts)

        # Crowd detection
        if avg_people_count > CROWD_THRESHOLD:
            print("Crowd detected!")
            alert_message = "Crowd detected at Railway Station. Immediate attention required."
            send_alert(subject="Crowd Detected", body=alert_message)
            send_whatsapp_alert(alert_message)
            cv2.putText(frame, "Crowd Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check for unattended bags
        for bag_id, last_seen in bag_last_seen_time.items():
            elapsed_time = time.time() - last_seen
            if elapsed_time > UNATTENDED_THRESHOLD:
                print("Unattended bag detected!")
                alert_message = "Unattended bag detected at Railway Station. Immediate attention required."
                send_alert(subject="Unattended Bag Detected", body=alert_message)
                send_whatsapp_alert(alert_message)
                cv2.putText(frame, "Unattended Bag Detected!", (bag_trackers[bag_id][0], bag_trackers[bag_id][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Railway Station Safety Monitoring", frame)
    else:
        print("No valid indices found for this frame.")

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
