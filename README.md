# 🎥 Advanced Video Analysis System

A sophisticated video analysis tool that leverages state-of-the-art Vision-Language Models (VLMs) to provide detailed, frame-by-frame analysis of video content. Perfect for fitness analysis, sports technique evaluation, action recognition, and general video understanding.

## ✨ Features

- **🧠 Advanced VLM Models**: Utilizes cutting-edge models like Llama Vision 90B for superior analysis quality
- **⚡ Multiple Analysis Modes**: Choose between detailed, standard, or fast analysis based on your needs
- **🎯 Specialized Analysis Types**: Action detection, detailed technical analysis, and scene understanding
- **📊 Comprehensive Reports**: Generate structured summaries with timestamps and technical insights
- **🔄 Fallback System**: Multiple model redundancy ensures reliable results
- **💾 Export Capabilities**: Save detailed analysis reports in JSON format
- **⏱️ Timestamp Tracking**: Frame-level timestamp information for precise analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Together AI API key
- OpenCV compatible video files

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/advanced-video-analysis.git
   cd advanced-video-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file
   echo "TOGETHER_API_KEY=your_together_ai_api_key_here" > .env
   ```

4. **Run the analysis**
   ```bash
   python main.py
   ```

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
opencv-python>=4.8.0
Pillow>=10.0.0
tqdm>=4.65.0
python-dotenv>=1.0.0
together>=0.2.7
```

## 🎮 Usage

### Basic Usage

```python
from video_analyzer import analyze_video_advanced

# Analyze a video with default settings
results = analyze_video_advanced("your_video.mp4")
print(results['summary'])
```

### Advanced Usage

```python
# Detailed analysis for fitness/sports videos
results = analyze_video_advanced(
    video_path="workout_video.mp4",
    mode="detailed",
    analysis_type="detailed_analysis"
)

# Quick action detection
results = analyze_video_advanced(
    video_path="action_video.mp4",
    mode="fast",
    analysis_type="action_detection"
)

# Scene understanding for general content
results = analyze_video_advanced(
    video_path="general_video.mp4",
    mode="standard",
    analysis_type="scene_understanding"
)
```

## ⚙️ Configuration

### Analysis Modes

| Mode | Frame Interval | Max Frames | Best For |
|------|---------------|------------|----------|
| **Detailed** | 15 frames | 20 | Technical analysis, form checking |
| **Standard** | 30 frames | 15 | General purpose analysis |
| **Fast** | 60 frames | 10 | Quick overview, action detection |

### Analysis Types

- **`action_detection`**: Focus on movements, actions, and activities
- **`detailed_analysis`**: Comprehensive technical analysis with form evaluation
- **`scene_understanding`**: Environmental context and object detection

## 🏗️ Architecture

### Core Components

1. **Frame Extraction**: Intelligent sampling of video frames with timestamp tracking
2. **VLM Analysis**: Multi-model analysis using state-of-the-art vision-language models
3. **Summary Generation**: AI-powered synthesis of frame analyses into coherent narratives
4. **Report Export**: Structured data export for further processing

### Supported Models

- **Llama-Vision-Free**: Latest free-tier Llama vision model
- **Llama-3.2-90B-Vision-Instruct-Turbo**: High-performance 90B parameter model
- **Llama-3.2-11B-Vision-Instruct-Turbo**: Balanced speed and quality option

## 📁 Project Structure

```
advanced-video-analysi/
├── main.py                 # Main analysis script
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── frames/                # Temporary frame storage (auto-created)
├── video_analysis_report.json               # Analysis reports (auto-created)
└── README.md             # This file
```

## 🎯 Use Cases

### Fitness & Sports Analysis
- **Form Analysis**: Evaluate exercise technique and posture
- **Progress Tracking**: Compare performance across sessions
- **Technique Improvement**: Identify areas for improvement

### Content Analysis
- **Action Recognition**: Identify and categorize activities
- **Scene Understanding**: Analyze environments and contexts
- **Quality Assessment**: Evaluate video content quality

### Research Applications
- **Behavioral Analysis**: Study movement patterns and behaviors
- **Activity Classification**: Automated activity labeling
- **Temporal Analysis**: Track changes over time

## 📊 Output Format

### Summary Output
```python
{
    'summary': 'Comprehensive narrative description...',
    'frame_analyses': [
        {
            'frame_number': 0,
            'timestamp': 0.0,
            'analysis': 'Detailed frame analysis...',
            'model_used': 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo'
        }
    ],
    'total_frames_analyzed': 15
}
```

### JSON Report Structure
- Frame-by-frame analysis with timestamps
- Model attribution for transparency
- Comprehensive video summary
- Technical metadata

## 🔧 Configuration Options

### Environment Variables
```bash
TOGETHER_API_KEY=your_api_key_here
```

### Advanced Configuration
```python
ANALYSIS_CONFIG = {
    "detailed": {
        "frame_interval": 15,    # Extract every 15th frame
        "batch_size": 2,         # Process 2 frames at once
        "max_tokens": 500        # Maximum response length
    }
}
```

## 🚨 Error Handling

The system includes robust error handling:
- **Model Fallback**: Automatically tries alternative models if primary fails
- **Frame Extraction**: Handles corrupted or unreadable video files
- **API Rate Limiting**: Built-in delays to prevent API overuse
- **Memory Management**: Automatic cleanup of temporary files

## 💰 Cost Considerations

- **Frame Sampling**: Intelligent frame selection to minimize API calls
- **Model Efficiency**: Automatic selection of most cost-effective models
- **Batch Processing**: Optimized request batching where possible

### Estimated Costs (Together AI)
- **Fast Mode**: ~$0.10-0.50 per video
- **Standard Mode**: ~$0.50-1.50 per video  
- **Detailed Mode**: ~$1.50-5.00 per video

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Together AI** for providing access to state-of-the-art VLM models
- **Meta AI** for the Llama vision models
- **OpenCV** community for video processing capabilities

## 🗺️ Roadmap

- [ ] **Real-time analysis** support
- [ ] **Batch video processing** capabilities
- [ ] **Custom model fine-tuning** options
- [ ] **Web interface** for easier usage
- [ ] **Integration** with popular video platforms
- [ ] **Mobile app** development

---

⭐ **Star this repository if you find it helpful!**
