# YOLO Speed Benchmark: Object Detection Performance Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)]()

A comprehensive benchmarking study comparing YOLOv5, YOLOv8, and YOLOv11 architectures for efficient object detection on edge devices.

## 📋 Project Overview

This project systematically evaluates the performance trade-offs between different YOLO versions to identify optimal models for deployment on resource-constrained devices like Raspberry Pi, Jetson Nano, and mobile platforms.

### Research Questions
- Which YOLO architecture offers the best speed vs. accuracy trade-off?
- How do different model sizes (nano, small, medium) perform on edge devices?
- What are the practical implications for real-time applications?

## 🎯 Objectives

- **Benchmark** YOLOv5, YOLOv8, and YOLOv11 architectures
- **Compare** nano, small, and medium model variants
- **Measure** inference speed (FPS), accuracy (mAP), latency, and memory usage
- **Analyze** performance on standard datasets (COCO val2017)
- **Recommend** optimal models for different deployment scenarios

## 🔬 Models to Benchmark

| YOLO Version | Model Variants |
|--------------|----------------|
| YOLOv5       | v5n, v5s, v5m  |
| YOLOv8       | v8n, v8s, v8m  |
| YOLOv11      | v11n, v11s, v11m |

## 📊 Performance Metrics

- **FPS (Frames Per Second)**: Inference throughput
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: COCO standard metric
- **Latency**: Time per image (milliseconds)
- **Model Size**: Memory footprint (MB)
- **GPU Memory**: VRAM usage during inference

## 🛠️ Technology Stack

- **Framework**: PyTorch
- **YOLO Implementation**: Ultralytics
- **Dataset**: COCO val2017
- **Hardware**: NVIDIA GPU (primary), CPU (comparison)
- **Tools**: Python, OpenCV, Matplotlib

## 📁 Project Structure

```
YOLO-Speed-Benchmark/
├── README.md
├── requirements.txt
├── benchmark.py           # Main benchmarking script
├── models/               # Model loading utilities
│   └── load_models.py
├── utils/                # Helper functions
│   ├── metrics.py
│   └── visualization.py
├── data/                 # Dataset handling
│   └── prepare_coco.py
├── results/              # Benchmark results (generated)
│   ├── metrics.csv
│   └── plots/
└── notebooks/            # Analysis notebooks
	└── analysis.ipynb
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
8GB+ RAM
```

### Installation
```bash
# Clone repository
git clone https://github.com/123sailee/YOLO-Speed-Benchmark.git
cd YOLO-Speed-Benchmark

# Install dependencies
pip install -r requirements.txt

# Download COCO val2017 dataset (optional - for full testing)
# Instructions in data/prepare_coco.py
```

### Quick Start
```bash
# Run benchmark on sample images
python benchmark.py --models yolov5n yolov8n yolov11n --samples 100

# Run full benchmark on COCO validation set
python benchmark.py --dataset coco --all-models
```

## 📈 Expected Results

Preliminary testing suggests:
- **Fastest**: YOLOv5n (~140+ FPS on RTX 3060)
- **Most Accurate**: YOLOv11m (~60+ mAP@0.5)
- **Best Balance**: YOLOv8n (speed + accuracy for edge devices)

*Full results will be documented in `results/` directory*

## 🔍 Current Status

🚧 **In Progress** - Active Development

**Completed:**
- [x] Project planning and research design
- [x] Repository setup and documentation
- [x] Technology stack selection
- [ ] Model loading pipeline
- [ ] Benchmarking script implementation
- [ ] COCO dataset integration
- [ ] Metrics collection and analysis
- [ ] Results visualization
- [ ] Final report and recommendations

## 📝 Methodology

1. **Model Loading**: Load pre-trained YOLO models using Ultralytics
2. **Dataset Preparation**: Use COCO val2017 (5000 images)
3. **Inference Testing**: Run each model on identical image set
4. **Metrics Collection**: Record FPS, mAP, latency for each model
5. **Comparative Analysis**: Generate comparison tables and visualizations
6. **Edge Testing**: Evaluate on resource-constrained hardware (if available)

## 🎓 Applications

- **Edge AI**: Optimal model selection for embedded systems
- **Real-time Detection**: Surveillance, autonomous vehicles, robotics
- **Mobile Deployment**: Object detection on smartphones
- **Resource Planning**: Understanding compute vs. accuracy trade-offs

## 🔗 Related Work

- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- [COCO Dataset](https://cocodataset.org/)

## 👤 Author

**Sailee Abhale**
- GitHub: [@123sailee](https://github.com/123sailee)
- Email: sailee2303@gmail.com
- Institution: Sanjivani University, Kopargaon

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Ultralytics team for YOLO implementations
- COCO dataset creators
- PyTorch community

---

**Last Updated**: November 2024  
**Status**: 🚧 Active Development
