import cv2
from deepface import DeepFace

# Define the emotion detection function
def get_emotion_from_face(frame):
    try:
        # Analyze the frame and get the dominant emotion
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print("Error in emotion detection:", e)
        return "neutral"  # Default to neutral if detection fails

