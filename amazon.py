import boto3
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize AWS Rekognition client
aws_region = 'eu-central-1'  # Replace with your AWS region
rekognition_client = boto3.client('rekognition', region_name=aws_region)

# Load the dataset
dataset = [
    {"file_path": "./photos/happy.jpg", "label": "Happy"},
    {"file_path": "./photos/sad.jpg", "label": "Sad"},
    {"file_path": "./photos/3.jpg", "label": "Fear"},
    {"file_path": "./photos/4.jpg", "label": "Angry"},
    {"file_path": "./photos/5.jpg", "label": "Calm"},
    {"file_path": "./photos/6.jpg", "label": "Angry"},
    {"file_path": "./photos/7.png", "label": "Confused"},
    {"file_path": "./photos/8.jpg", "label": "Angry"},
    {"file_path": "./photos/9.jpg", "label": "Fear"},
    {"file_path": "./photos/10.jpg", "label": "Calm"},
    {"file_path": "./photos/11.jpg", "label": "Sad"},
    {"file_path": "./photos/12.jpg", "label": "Happy"},
    {"file_path": "./photos/13.jpg", "label": "Fear"},
    {"file_path": "./photos/14.jpg", "label": "Confused"},
    {"file_path": "./photos/15.jpg", "label": "Surprised"},
    {"file_path": "./photos/16.jpg", "label": "Disgusted"},
    {"file_path": "./photos/17.jpg", "label": "Disgusted"},
    {"file_path": "./photos/18.jpg", "label": "Disgusted"},
    {"file_path": "./photos/19.jpg", "label": "Angry"},
    {"file_path": "./photos/20.jpg", "label": "Happy"},
    {"file_path": "./photos/21.jpg", "label": "Fear"},
]

# Mapping between Rekognition emotions and dataset labels
emotion_mapping = {
    "HAPPY": "Happy",
    "SAD": "Sad",
    "FEAR": "Fear",
    "ANGRY": "Angry",
    "CALM": "Calm",
    "CONFUSED": "Confused",
    "SURPRISED": "Surprised",
    "DISGUSTED": "Disgusted",
    "Not specified": "Not specified",
}

# Initialize lists to store actual and predicted labels
actual_labels = []
predicted_labels = []

# Iterate through the dataset
for entry in dataset:
    file_path = entry['file_path']
    label = entry['label']

    # Analyze the image with Rekognition
    try:
        with open(file_path, 'rb') as image_file:
            image_bytes = image_file.read()

        response = rekognition_client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )

        # Check if any faces are detected
        if response['FaceDetails']:
            # Extract emotions detected by Rekognition along with their confidence scores
            emotions = [(emotion['Type'], emotion['Confidence']) for face in response['FaceDetails'] for emotion in face['Emotions']]

            # Select the emotion with the highest confidence
            primary_emotion, confidence = max(emotions, key=lambda x: x[1])
        else:
            # If no faces are detected, return "Not specified"
            primary_emotion = "Not specified"
            confidence = 0.0

        # Map the Rekognition emotion to dataset label
        mapped_label = emotion_mapping.get(primary_emotion, "Not specified")

        # Append actual and predicted labels
        actual_labels.append(label)
        predicted_labels.append(mapped_label)

        # Print actual and predicted labels along with confidence
        print(f"Actual Label: {label}")
        print(f"Predicted Label: {mapped_label} (Confidence: {confidence:.2f})")
        print()

    except Exception as e:
        # Handle any exceptions (e.g., image not found)
        print(f"Error analyzing image {file_path}: {str(e)}")

# Calculate Precision, Recall, and F1 Score using scikit-learn
precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)

# Print the metrics
print("Amazon Rekognition Metrics:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
