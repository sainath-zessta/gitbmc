#  TimeTracker.py
import cv2
from ultralytics import YOLO
import os
import base64
import requests
import shutil


class TimeStampExtractor:
    def __init__(self, video_path, output_dir='output/frames/'):
        api_key = os.getenv('OPENAI_API_KEY')
        self.model = YOLO('Models/timestamp.pt')
        self.api_key = api_key
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_first_detection(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_saved = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_saved:
                break
            
            # Perform inference on the frame
            results = self.model(frame)
            
            # Extract bounding boxes
            bboxes = results[0].boxes.xyxy
            
            # Check if only one object is detected
            if len(bboxes) == 1:
                # Crop the frame to the bounding box of the detected object
                x1, y1, x2, y2 = map(int, bboxes[0])
                cropped_frame = frame[y1:y2, x1:x2]
                
                # Save the cropped frame
                frame_filename = os.path.join(self.output_dir, 'detected_object.jpg')
                cv2.imwrite(frame_filename, cropped_frame)
                print(f'Saved: {frame_filename}')
                
                # Set the flag to True to ensure only one frame is saved
                frame_saved = True

        cap.release()
        return frame_filename if frame_saved else None

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_timestamp(self, image_path):
        base64_image = self.encode_image(image_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        schema = """HH:MM:SS"""

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Extract date and time using your multimodal capabilities and return reponse with respect to given {schema}, If it is a blank image return '00:00:00' and don't give any code, notations and English explanation"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            
            response=response.json()
            content = response.get('choices')[0].get("message").get('content')
            print("**********",content)
            return content
        except Exception as ex:
            print("Exception: ------> In TIMETRACKER", ex)
            return None

    def cleanup(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print(f"Cleaned up: {self.output_dir}")

    def run(self):
        frame_path = self.extract_first_detection()
        if frame_path:
            return self.extract_timestamp(frame_path)
        else:
            self.cleanup()
            return None
        

def detect_time(video_path):
    return TimeStampExtractor(video_path).run()
    
