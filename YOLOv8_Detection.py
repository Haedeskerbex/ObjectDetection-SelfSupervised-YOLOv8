import cv2
import supervision as sv
from ultralytics import YOLO
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import threading
import time
import os

# Authenticate and initialize Google Drive access
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Opens a web browser for authentication
drive = GoogleDrive(gauth)

# Set paths and parameters
shared_model_path = "latest_best.pt"
local_frames_path = "captured_frames"  # Local directory to save frames
google_drive_folder_id = "1s7JqFs_vb7IoYe6S0VL4wOFonK8eqgdS"  # Replace with your Google Drive folder ID
file_id = '1T_7Bc8XZj7Te5ufzTnonAXqQ23-SR_4F'  # File ID for latest_best.pt on Google Drive

# Create a local directory for frames if it doesn't exist
if not os.path.exists(local_frames_path):
    os.makedirs(local_frames_path)

# A thread-safe queue to hold files to be uploaded
upload_queue = []

def download_model(file_id, destination):
    """Download model file if updated."""
    file = drive.CreateFile({'id': file_id})
    file.FetchMetadata()  # Fetch metadata to check for updates
    file_modified_time = file['modifiedDate']

    # Check if the model file exists locally
    if os.path.exists(destination):
        local_modified_time = time.ctime(os.path.getmtime(destination))
    else:
        local_modified_time = None

    # Download if the remote file is newer or if it doesn't exist locally
    if local_modified_time is None or file_modified_time > local_modified_time:
        print("Downloading the updated model...")
        file.GetContentFile(destination)
        print("Model updated successfully.")
    else:
        print("Local model is up-to-date.")

def model_update_worker():
    """Worker function to periodically check and update the model."""
    while True:
        download_model(file_id, shared_model_path)
        time.sleep(60) 

def upload_to_drive(file_path):
    """Upload a file to Google Drive."""
    file_name = os.path.basename(file_path)
    file_drive = drive.CreateFile({
        'title': file_name,
        'parents': [{'id': google_drive_folder_id}]  # Specify the folder in Drive
    })
    file_drive.SetContentFile(file_path)
    file_drive.Upload()
    print(f"Uploaded {file_name} to Google Drive.")

def upload_worker():
    """Worker function to handle file uploads."""
    while True:
        if upload_queue:  # Check if there are files to upload
            file_to_upload = upload_queue.pop(0)  # Get the next file to upload
            upload_to_drive(file_to_upload)
        time.sleep(1)  # Sleep briefly to avoid busy waiting

def capture_frames_and_detect():
    """Capture frames from the webcam and perform detection."""
    model = YOLO(shared_model_path)
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    detection_count = 0
    detection_threshold = 5  # Set initial detection threshold

    while True:
        key = cv2.waitKey(1)

        # Check for model update
        new_modified_time = os.path.getmtime(shared_model_path)
        if new_modified_time != os.path.getmtime(shared_model_path):
            print("New model detected, reloading...")
            model = YOLO(shared_model_path)  # Reload the updated model

        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # If the frame was not read successfully, break the loop
        if not ret:
            break
        
        elif key == ord('q'):
            print("Quitting...")
            break

        results = model(frame, conf=0.35)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate the frame
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)       
        cv2.imshow("Webcam", annotated_image)

        # Check for detections
        if len(detections) > 0:
            detection_count += 1
            # Save the frame to the local directory if detections exceed the threshold
            frame_filename = f"{local_frames_path}/frame_{int(time.time())}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Detection {detection_count}: Frame saved to {frame_filename}")

            # Add the file to the upload queue
            upload_queue.append(frame_filename)

        # Adjust threshold based on detection frequency
        if detection_count >= detection_threshold:
            print("Increasing detection threshold.")
            detection_threshold += 1  # Increase threshold as needed
        elif detection_count < detection_threshold - 2:  # Decrease if detections are sparse
            print("Decreasing detection threshold.")
            detection_threshold = max(1, detection_threshold - 1)  # Ensure it doesn't go below 1

    # Release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the upload worker and model update in separate threads
upload_thread = threading.Thread(target=upload_worker)
upload_thread.daemon = True
upload_thread.start()

model_update_thread = threading.Thread(target=model_update_worker)
model_update_thread.daemon = True
model_update_thread.start()

# Run the capture and detection in a separate thread
detection_thread = threading.Thread(target=capture_frames_and_detect)
detection_thread.start()




