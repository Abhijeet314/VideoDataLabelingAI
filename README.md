# Video Description Generator

An AI-powered tool that automatically generates detailed descriptions of videos by extracting frames, analyzing them with computer vision, and creating coherent summaries using large language models.

## Features

- **Frame Extraction**: Automatically extracts frames from video files at configurable intervals
- **Image Captioning**: Uses Vision Transformer (ViT) + GPT-2 model to generate captions for each frame
- **Intelligent Summarization**: Leverages Meta's Llama-3 70B model to create coherent video descriptions
- **Batch Processing**: Processes multiple frames efficiently in batches
- **GPU Support**: Automatically detects and uses CUDA if available for faster processing

## How It Works

1. **Video Frame Extraction**: The script extracts frames from the input video at regular intervals (default: every 30th frame)
2. **Image Analysis**: Each extracted frame is processed through a pre-trained ViT-GPT2 model to generate descriptive captions
3. **Batch Processing**: Frames are processed in batches of 4 for optimal performance
4. **Caption Aggregation**: All individual frame captions are combined into a single text
5. **AI Summarization**: The combined captions are sent to Llama-3 70B via Together AI API to create a coherent, non-repetitive video description

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster processing)
- Together AI API key

## Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd video-description-generator
```

2. **Install required dependencies**:
```bash
pip install opencv-python
pip install transformers
pip install torch torchvision
pip install Pillow
pip install tqdm
pip install python-dotenv
pip install together
```

3. **Set up environment variables**:
Create a `.env` file in the project root and add your Together AI API key:
```
TOGETHER_API_KEY=your_together_ai_api_key_here
```

## Usage

1. **Place your video file** in the project directory
2. **Update the video filename** in the script:
```python
video_file = "your_video_file.avi"  # Change this to your video file
```
3. **Run the script**:
```bash
python video_description_generator.py
```

## Configuration

You can customize the following parameters in the script:

- `frame_interval`: Controls how often frames are extracted (default: 30 = every 30th frame)
- `max_length`: Maximum length for generated captions (default: 16 tokens)
- Batch size for frame processing (default: 4 frames per batch)

## Dependencies

### Core Libraries
- **OpenCV (`cv2`)**: Video processing and frame extraction
- **Transformers**: Hugging Face library for the ViT-GPT2 model
- **PyTorch**: Deep learning framework for model inference
- **PIL (Pillow)**: Image processing and format conversion

### AI Models
- **Vision Encoder-Decoder Model**: `nlpconnect/vit-gpt2-image-captioning`
  - Vision Transformer (ViT) for image encoding
  - GPT-2 for caption generation
- **Together AI**: Meta-Llama-3-70B-Instruct-Turbo for text summarization

### Utility Libraries
- **tqdm**: Progress bars for batch processing
- **python-dotenv**: Environment variable management
- **together**: Together AI API client

## Output

The script will:
1. Extract frames from your video and save them temporarily in a `frames/` folder
2. Display progress as it processes batches of frames
3. Generate individual captions for each frame
4. Create a final coherent video description using AI summarization
5. Print the final description to the console

## Example Output

```
120 frames extracted.
Processing frames: 100%|██████████| 30/30 [02:15<00:00,  4.50s/it]

Final Video Description:
The video shows a soccer penalty kick scenario taking place on a grass field. A player in a dark jersey approaches the ball positioned at the penalty spot while a goalkeeper in a bright colored uniform prepares to defend the goal. The scene captures the tense moment before the kick, with the field markings clearly visible and the goal posts framing the action.
```

## Notes

- The script automatically cleans up extracted frames after processing
- GPU acceleration is used automatically if CUDA is available
- The Together AI API requires an active subscription for the Llama-3 model
- Processing time depends on video length and hardware capabilities

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use CPU processing
- **API errors**: Verify your Together AI API key is correct and active
- **Video format issues**: Ensure your video file is in a supported format (MP4, AVI, MOV, etc.)
- **Missing dependencies**: Run `pip install -r requirements.txt` if you create a requirements file

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
