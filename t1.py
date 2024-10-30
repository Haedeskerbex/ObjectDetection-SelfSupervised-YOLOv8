import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('efficientdet_trained_model.h5')

# Function to preprocess the frames
def preprocess_frame(frame):
    # Resize frame to the input size of the model (assumed to be 512x512)
    resized_frame = cv2.resize(frame, (512, 512))
    # Normalize the frame (convert values to float32 and scale to [0,1])
    resized_frame = resized_frame.astype(np.float32) / 255.0
    # Add batch dimension (1, 512, 512, 3)
    input_frame = np.expand_dims(resized_frame, axis=0)
    return input_frame

# Function to draw bounding boxes and labels
def draw_detections(frame, boxes, scores, labels, threshold=0.5):
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]
            label = labels[i]
            score = scores[i]

            # Scale box coordinates back to the original frame size
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw the label and score
            label_text = f'{label}: {score:.2f}'
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Open a connection to the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Run the frame through the EfficientDet model
    predictions = model.predict(input_frame)

    # Get the predictions (boxes, scores, and labels)
    boxes = predictions['boxes'][0]  # Shape (num_boxes, 4)
    scores = predictions['scores'][0]  # Shape (num_boxes,)
    labels = predictions['labels'][0]  # Shape (num_boxes,)

    # Draw detections on the frame
    draw_detections(frame, boxes, scores, labels, threshold=0.5)

    # Display the frame with the detections
    cv2.imshow('Webcam Object Detection', frame)

    # Press 'q' to exit the loop and close the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
