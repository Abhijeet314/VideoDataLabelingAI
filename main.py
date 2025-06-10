import cv2
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from together import Together

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
frame_interval = 30  
gen_kwargs = {"max_length": max_length}

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def extract_frames(video_path, output_folder="frames", interval=frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.png")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
        frame_count += 1

    cap.release()
    return saved_frames

def generate_video_description(video_path):
    frames = extract_frames(video_path)
    print(f"{len(frames)} frames extracted.")

    all_captions = []
    for i in tqdm(range(0, len(frames), 4)): 
        batch = frames[i:i+4]
        captions = predict_step(batch)
        all_captions.extend(captions)

    combined_description = " ".join(all_captions)
    return combined_description


video_file = "SoccerPenalty.avi" 
description = generate_video_description(video_file)

def get_output_llm(text):
    prompt = f"""
    The following text is a list of image captions extracted from a video. The text is repetitive and not well-structured.
    Please rewrite it into a concise and coherent summary without repetition:

    Captions:
    {text}
"""
    client = Together()

    response = client.chat.completions.create(
         model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        messages=[
        {
            "role": "user",
            "content": prompt
        }
        ]
    )

    return response.choices[0].message.content
    
response = get_output_llm(description)
print("\nFinal Video Description:\n", response)
