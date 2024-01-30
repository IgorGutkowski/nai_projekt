import os
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
from sklearn.metrics import precision_score, recall_score, f1_score

# Path to your service account JSON key file
service_account_file = './api.json'

# Authenticate the client using your service account JSON key file
credentials = service_account.Credentials.from_service_account_file(service_account_file)
client = vision.ImageAnnotatorClient(credentials=credentials)

# Adjusted dataset (example)
dataset = [
    {"file_path": "./photos/happy.jpg", "label": "Joy"},
    {"file_path": "./photos/sad.jpg", "label": "Sorrow"},
    {"file_path": "./photos/3.jpg", "label": "Sorrow"},
    {"file_path": "./photos/4.jpg", "label": "Anger"},
    {"file_path": "./photos/5.jpg", "label": "Joy"},
    {"file_path": "./photos/6.jpg", "label": "Anger"},
    {"file_path": "./photos/7.png", "label": "Sorrow"},
    {"file_path": "./photos/8.jpg", "label": "Anger"},
    {"file_path": "./photos/9.jpg", "label": "Surprise"},
    {"file_path": "./photos/10.jpg", "label": "Joy"},
    {"file_path": "./photos/11.jpg", "label": "Sorrow"},
    {"file_path": "./photos/12.jpg", "label": "Joy"},
    {"file_path": "./photos/13.jpg", "label": "Surprise"},
    {"file_path": "./photos/14.jpg", "label": "Sorrow"},
    {"file_path": "./photos/15.jpg", "label": "Surprise"},
    {"file_path": "./photos/16.jpg", "label": "Sorrow"},
    {"file_path": "./photos/17.jpg", "label": "Sorrow"},
    {"file_path": "./photos/18.jpg", "label": "Anger"},
    {"file_path": "./photos/19.jpg", "label": "Anger"},
    {"file_path": "./photos/20.jpg", "label": "Joy"},
    {"file_path": "./photos/21.jpg", "label": "Surprise"},
    {"file_path": "./photos/white.jpg", "label": "Not specified"},



]

# Initialize a list to store detected emotions for each image
detected_emotions = []

# Initialize lists to store actual and predicted labels
actual_labels = []
predicted_labels = []

# Iterate through the dataset
for entry in dataset:
    file_path = entry['file_path']
    label = entry['label']

    try:
        with open(file_path, 'rb') as image_file:
            image_bytes = image_file.read()

        response = client.face_detection(image={'content': image_bytes})
        faces = response.face_annotations

        if faces:
            primary_face = faces[0]  # Assuming the first face is the primary face

            # Store detected emotions for the primary face
            emotions = {
                'Joy': primary_face.joy_likelihood,
                'Sorrow': primary_face.sorrow_likelihood,
                'Anger': primary_face.anger_likelihood,
                'Surprise': primary_face.surprise_likelihood
            }

            # Find the emotion with the highest likelihood
            detected_emotion = max(emotions, key=emotions.get)

            detected_emotions.append(detected_emotion)
            actual_labels.append(label)
            predicted_labels.append(detected_emotion)

            # Print the actual label and detected emotion
            print(f"File: {file_path}")
            print(f"Actual Label: {label}")
            print(f"Detected Emotion: {detected_emotion}")
            print()  # Add a line break for readability

        else:
            print(f"No faces detected in the image: {file_path}")

    except Exception as e:
        print(f"Error analyzing image with Google Cloud Vision {file_path}: {str(e)}")

# Calculate Precision, Recall, and F1 Score using scikit-learn
actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)
precision_vision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
recall_vision = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
f1_vision = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)

# Print the metrics for Google Cloud Vision
print("\nGoogle Cloud Vision Metrics:")
print(f"Precision: {precision_vision}")
print(f"Recall: {recall_vision}")
print(f"F1 Score: {f1_vision}")
