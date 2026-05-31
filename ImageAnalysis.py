

import cv2
from pathlib import Path


from openai import OpenAI
import anthropic

from google import genai
import base64
import os
from PIL import Image
import io
import json


from google.genai import types
from pydantic import BaseModel, Field

# Load existing results if file exists
if os.path.exists("image_analysis.json"):
    with open("image_analysis.json", "r") as f:
        all_results = json.load(f)
else:
    all_results = {}


openai_client = OpenAI()
claude_client = anthropic.Anthropic()
gemini_client = genai.Client()

def extract_frames(video_path, output_dir="frames", every_n_seconds=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * every_n_seconds)

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % interval == 0:
            frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    return frame_paths


video_path = "Gesture 21.mp4"
frames = extract_frames(video_path, every_n_seconds=1)
print(frames[:10])




def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Use only first few frames first to control cost
selected_frames = frames[:10]


# OpenAI API

content = [
    {
        "type": "input_text",
        "text": """You are an expert in kinesics and body language. 
            Analyze this sequential collection of images which represent frames extracted chronologically from a video.
            
            Context:
            The primary meaning, intent, or cultural context of this gesture.

            Question:
            Given that the person in this video is Japanese, what does the hand gesture mean?

            Return only:
            Likely meaning:
            Confidence:
            """
    }
]

for frame_path in selected_frames:
    base64_image = encode_image(frame_path)
    content.append({
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{base64_image}",
        "detail": "low"
    })

gpt_response = openai_client.responses.create(
    model="gpt-5.5",
    input=[
        {
            "role": "user",
            "content": content
        }
    ]
)
print(gpt_response.output_text)


# Claude API

content_payload = []

for i, filename in enumerate(selected_frames):
    with Image.open(filename) as img:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    content_payload.append({"type": "text", "text": f"Frame {i+1}:"})
    content_payload.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_base64,
        }
    })

content_payload.append({
    "type": "text",
    "text": """You are an expert in kinesics and body language. 
            Analyze this sequential collection of images which represent frames extracted chronologically from a video.
        
            Context:
            The primary meaning, intent, or cultural context of this gesture.

            Question:
            Given that the person in this video is Japanese, what does the hand gesture mean?

            Return only:
            Likely meaning:
            Confidence:
            """
})

claude_message = claude_client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    messages=[{"role": "user", "content": content_payload}]
)

print(claude_message.content[0].text)

# Google Gemini

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 2. Define the JSON structure you want back
class GestureAnalysis(BaseModel):
    meaning: str = Field(description="The primary meaning, intent, or cultural context of this gesture.")



# 4. Prepare the contents array
# We start with the prompt instructions...
contents_payload = [
    """You are an expert in kinesics and body language. 
    Analyze this sequential collection of images which represent frames extracted chronologically from a video.
    Given that the person in this video is Japanese, what does the hand gesture mean?"""
]

# ...then we loop through your selected frames, encode them, 
# and append them as inline image data parts
for frame_path in selected_frames:
    encoded_str = encode_image(frame_path)
    image_part = types.Part.from_bytes(
        data=base64.b64decode(encoded_str),
        mime_type="image/jpeg" # Change to image/png if your frames are PNGs
    )
    contents_payload.append(image_part)

# 5. Send everything to Gemini
# print(f"Sending {len(selected_frames)} frames to Gemini for analysis...")

gemini_response = gemini_client.models.generate_content(
    model='gemini-2.5-flash',
    contents=contents_payload,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GestureAnalysis,
        temperature=0.2
    ),
)

# 6. Output the structured answer
print("\n--- Gesture Analysis Results ---")
print(gemini_response.text)





# Add new results
all_results[video_path] = {
    "frames": len(selected_frames),
    "GPT analysis": gpt_response.output_text,
    "Claude analysis": claude_message.content[0].text,
    "Gemini analysis": gemini_response.text,
    "video": video_path
}

# Write everything back
with open("image_analysis.json", "w") as f:
    json.dump(all_results, f, indent=2)