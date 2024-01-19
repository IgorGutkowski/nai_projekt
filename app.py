from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import boto3
import os
import time

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No file selected'), 400

    file_content = file.read()  # Read the file content once and store it

    aws_region = 'eu-central-1'
    rekognition_client = boto3.client('rekognition', region_name=aws_region)
    polly_client = boto3.client('polly', region_name=aws_region)

    try:
        response = rekognition_client.detect_faces(
            Image={'Bytes': file_content},
            Attributes=['ALL']
        )

        # Check if any faces are detected
        if not response['FaceDetails']:
            # No faces detected, so synthesize speech for "Not Specified"
            polly_response = polly_client.synthesize_speech(
                Text="The primary emotion is not specified",
                OutputFormat='mp3',
                VoiceId='Joanna'
            )
            
            # Use the current timestamp as a unique identifier for the audio file
            timestamp = int(time.time())
            audio_filename = f'polly_audio.mp3'
            audio_filepath = os.path.join(app.static_folder, audio_filename)
            
            # Write the synthesized speech to the audio file
            with open(audio_filepath, 'wb') as audio_file:
                audio_file.write(polly_response['AudioStream'].read())
            
            # Construct the unique audio file URL
            audio_url = request.url_root + f'audio/{audio_filename}'

            # Return response with "Not Specified" emotion
            return jsonify(emotion="Not Specified", audioUrl=audio_url), 200

        # If faces are detected, find the primary face and its emotion
        primary_face = max(response['FaceDetails'], key=lambda x: x['BoundingBox']['Width'] * x['BoundingBox']['Height'])
        primary_emotion = max(primary_face['Emotions'], key=lambda x: x['Confidence'])['Type']

        # Synthesize speech for the primary emotion
        polly_response = polly_client.synthesize_speech(
            Text=f"The primary emotion is {primary_emotion}",
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        
        # Use the current timestamp as a unique identifier for the audio file
        timestamp = int(time.time())
        audio_filename = f'polly_audio.mp3'
        audio_filepath = os.path.join(app.static_folder, audio_filename)
        
        # Write the synthesized speech to the audio file
        with open(audio_filepath, 'wb') as audio_file:
            audio_file.write(polly_response['AudioStream'].read())
        
        # Construct the unique audio file URL
        audio_url = request.url_root + f'audio/{audio_filename}'

        # Return the detected emotion and the bounding box of the primary face
        return jsonify(
            emotion=primary_emotion, 
            audioUrl=audio_url,
            boundingBox=primary_face['BoundingBox']
        ), 200

    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/audio/<filename>')
def audio(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
