import cv2
import os
import base64
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from together import Together
import json
import time

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)

ANALYSIS_CONFIG = {
    "detailed": {
        "frame_interval": 15, 
        "batch_size": 2,    
        "max_tokens": 500
    },
    "standard": {
        "frame_interval": 30,
        "batch_size": 4,
        "max_tokens": 300
    },
    "fast": {
        "frame_interval": 60,
        "batch_size": 6,
        "max_tokens": 200
    }
}

def encode_image_to_base64(image_path):
    """Convert image to base64 for API consumption"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_frames(video_path, output_folder="frames", interval=30):
    """Extract frames from video at specified intervals"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info: {total_frames} total frames, {fps:.2f} FPS")

    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_count % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            saved_frames.append({
                'path': frame_path,
                'timestamp': frame_count / fps,
                'frame_number': frame_count
            })
        frame_count += 1

    cap.release()
    return saved_frames

def analyze_frames_with_vlm(frame_paths, analysis_type="action_detection"):
    """Analyze frames using advanced VLM models from Together AI"""
    
    models_to_try = [
        "meta-llama/Llama-Vision-Free",         
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",  
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
    ]
    
    prompts = {
        "action_detection": """Analyze this video frame and describe:
1. What actions or movements are happening
2. The setting and environment
3. Any objects or equipment visible
4. The overall activity being performed

Be specific and detailed in your description.""",
        
        "detailed_analysis": """Provide a comprehensive analysis of this frame:
1. Main subject and their posture/position
2. Specific actions or exercises being performed
3. Technical details about form and technique
4. Environmental context and equipment
5. Any notable details about the movement or setup

Focus on accuracy and specific details.""",
        
        "scene_understanding": """Describe this video frame focusing on:
1. The overall scene and setting
2. Main activities happening
3. Key objects and their positions
4. The progression of movement or action
5. Any important contextual details

Provide a clear, descriptive summary."""
    }
    
    prompt = prompts.get(analysis_type, prompts["action_detection"])
    
    results = []
    
    for model_name in models_to_try:
        try:
            print(f"Trying model: {model_name}")
            
            for i, frame_info in enumerate(tqdm(frame_paths, desc=f"Analyzing with {model_name}")):
                try:
                    base64_image = encode_image_to_base64(frame_info['path'])
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ]
                    
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=400,
                        temperature=0.1  
                    )
                    
                    analysis = response.choices[0].message.content
                    
                    results.append({
                        'frame_number': frame_info['frame_number'],
                        'timestamp': frame_info['timestamp'],
                        'analysis': analysis,
                        'model_used': model_name
                    })
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error analyzing frame {i} with {model_name}: {e}")
                    continue
            
            if results:
                print(f"Successfully analyzed {len(results)} frames with {model_name}")
                break
                
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
    
    return results

def generate_comprehensive_summary(analysis_results, video_info=None):
    """Generate a comprehensive video summary using the analysis results"""
    
    # Combine all analyses
    combined_analysis = []
    for result in analysis_results:
        timestamp_str = f"[{result['timestamp']:.1f}s]"
        combined_analysis.append(f"{timestamp_str} {result['analysis']}")
    
    analysis_text = "\n".join(combined_analysis)
    
    prompt = f"""
Based on the following detailed frame-by-frame analysis of a video, create a comprehensive and coherent summary:

FRAME ANALYSES:
{analysis_text}

Please provide:
1. **Video Overview**: A concise summary of the main activity/content
2. **Key Actions**: The primary actions or exercises being performed
3. **Technical Details**: Any specific techniques, form, or methods observed
4. **Progression**: How the activity develops throughout the video
5. **Environment & Equipment**: Description of the setting and any equipment used

Format your response as a well-structured, flowing narrative that captures the essence of the video without repetition. Focus on the most important and consistent elements across the frames.
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating comprehensive summary."

def analyze_video_advanced(video_path, mode="standard", analysis_type="detailed_analysis"):
    """Main function to analyze video with advanced VLM"""
    
    config = ANALYSIS_CONFIG[mode]
    
    print(f"Starting advanced video analysis in {mode} mode...")
    print(f"Analysis type: {analysis_type}")
    
    # Extract frames
    frames = extract_frames(
        video_path, 
        interval=config["frame_interval"]
    )
    
    print(f"Extracted {len(frames)} frames for analysis")
    
    if not frames:
        return "No frames could be extracted from the video."
    
    max_frames = 20 if mode == "detailed" else 15 if mode == "standard" else 10
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
        print(f"Selected {len(frames)} frames for analysis")
    
    analysis_results = analyze_frames_with_vlm(frames, analysis_type)
    
    if not analysis_results:
        return "Failed to analyze video frames."
    
    final_summary = generate_comprehensive_summary(analysis_results)
    
    return {
        'summary': final_summary,
        'frame_analyses': analysis_results,
        'total_frames_analyzed': len(analysis_results)
    }

def save_analysis_report(results, output_file="video_analysis_report.json"):
    """Save detailed analysis report to file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed analysis report saved to {output_file}")

if __name__ == "__main__":
    video_file = "CuttingInKitchen.avi"
    
    
    print("Starting advanced video analysis...")
    results = analyze_video_advanced(
        video_file, 
        mode="detailed",
        analysis_type="detailed_analysis"
    )
    
    if isinstance(results, dict):
        print("\n" + "="*60)
        print("COMPREHENSIVE VIDEO ANALYSIS")
        print("="*60)
        print(results['summary'])
        print(f"\nAnalysis based on {results['total_frames_analyzed']} frames")
        
        save_analysis_report(results)
    else:
        print(f"Analysis result: {results}")