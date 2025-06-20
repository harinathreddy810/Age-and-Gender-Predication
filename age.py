import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained models for age and gender
age_model = r"D:\Harinathreddy\MY PROJECTS\age and gender\age_net.caffemodel"
age_proto = r"D:\Harinathreddy\MY PROJECTS\age and gender\age_deploy.prototxt"
gender_model = r"D:\Harinathreddy\MY PROJECTS\age and gender\gender_net.caffemodel"
gender_proto = r"D:\Harinathreddy\MY PROJECTS\age and gender\gender_deploy.prototxt"

# Load the models
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Mean values for model normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age groups & gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    # Display the results on the frame
    label = f"Gender: {gender}, Age: {age}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert the image from BGR (OpenCV default) to RGB (for matplotlib)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use matplotlib to display the image
    plt.imshow(frame_rgb)
    plt.title(label)
    plt.axis('off')  # Hide axes
    plt.pause(0.005)  # Pause to update the plot

    # Break the loop when 'q' is pressed
    if plt.waitforbuttonpress(timeout=0.01):
        break

# Release the capture
cap.release()
plt.close()
