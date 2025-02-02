# Vision2Motion

This ROS2 package provides nodes that analyze images using the PaliGemma language model and generate descriptions in both Japanese and English.

## Features

- Real-time image analysis using Google's PaliGemma-3B model
- Descriptions available in both Japanese and English
- GPU support for faster processing
- Supports USB cameras and any ROS2 image topics
- Debug image output with corresponding frame IDs

## Roadmap

This package is part of a larger project to enable vision-based robot control:

- ✅ Image analysis and description (Current)
- 🔄 Motion control nodes based on visual input (Planned)
- 🔄 Task planning based on visual understanding

The goal is to create a comprehensive framework for vision-guided robot motion control, 
enabling robots to understand their environment through vision and execute appropriate control tasks.

## Prerequisites

- ROS2 Jazzy
- NVIDIA GPU with CUDA support (optional, but recommended)
- USB camera (for live video input)

### System Dependencies

```bash
# Install ROS2 dependencies
sudo apt install ros-jazzy-cv-bridge ros-jazzy-usb-cam

# Install Python dependencies
pip3 install torch torchvision transformers pillow
```

### HuggingFace Token

You need to set up your HuggingFace access token. Add this to your `~/.bashrc`:

```bash
export HF_TOKEN="your_token_here"
```

Don't forget to source after adding:
```bash
source ~/.bashrc
```

## Installation

1. Create a workspace (if you don't have one):
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

2. Clone this repository:
```bash
git clone https://github.com/YourUsername/vision2motion.git
```

3. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select vision2motion
source install/setup.bash
```

## Usage

### Basic Usage with USB Camera

1. First, launch the USB camera node:
```bash
ros2 run usb_cam usb_cam_node_exe
```

2. In another terminal, start the image analysis node (Japanese):
```bash
ros2 launch vision2motion visual_explain_paligemma.launch.py hf_token:=$HF_TOKEN
```

Or for English descriptions:
```bash
ros2 launch vision2motion visual_explain_paligemma_en.launch.py hf_token:=$HF_TOKEN
```

3. Optional: View the camera feed using rqt_image_view:
```bash
ros2 run rqt_image_view rqt_image_view
```

### Topics

Japanese Node:
- Input: `/usb_cam/image_raw` (default)
- Output Text: `/picture_explanation`
- Debug Image: `/picture_explanation/debug_image`

English Node:
- Input: `/usb_cam/image_raw` (default)
- Output Text: `/picture_explanation_en`
- Debug Image: `/picture_explanation_en/debug_image`

### Customizing Input Source

You can change the input topic using the launch file parameter:
```bash
ros2 launch vision2motion visual_explain_paligemma.launch.py hf_token:=$HF_TOKEN input_topic:=/your/custom/topic
```

## Troubleshooting

### USB Camera Access
If you encounter permission issues with the USB camera:
```bash
sudo chmod 666 /dev/video0
```

### GPU Memory Issues
If you experience GPU out-of-memory errors, the node will automatically fall back to CPU. You can also adjust the GPU memory limit in the code:
```python
"max_memory": {0: "3GiB"}  # Adjust this value based on your GPU
```

## License

Apache-2.0 License (same as ROS2)

## Author

Michitoshi Tsubaki <michi.tsubaki.tech@gmail.com>

## Reference
- トランジスタ技術 2025.1.(Total Vol.724), 注目のロボットセンサ＆走行制御 pp.58-68, CQ出版社
