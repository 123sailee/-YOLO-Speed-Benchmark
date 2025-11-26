"""
YOLO Speed Benchmark - Main Benchmarking Script
Author: Sailee Abhale
Date: November 2024

This script benchmarks YOLOv5, YOLOv8, and YOLOv11 models for speed and accuracy.
"""

import argparse
import time
from pathlib import Path
import torch
import pandas as pd
from ultralytics import YOLO


class YOLOBenchmark:
    """
    Benchmark different YOLO models for speed and accuracy comparison.
    """

    def __init__(self, device: str = None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        print(f"Initialized benchmark on device: {self.device}")

    def load_model(self, model_name: str):
        """
        Load a YOLO model by name.

        Args:
            model_name (str): Model identifier (e.g., 'yolov5n', 'yolov8s')

        Returns:
            YOLO: Loaded model or None on error
        """
        try:
            print(f"Loading model: {model_name}...")
            # Expecting model files like 'yolov8n.pt' to exist, or model identifiers supported by Ultralytics
            model_path = f"{model_name}.pt"
            if not Path(model_path).exists():
                # Allow passing model_name directly to Ultralytics (it can fetch from hub)
                model = YOLO(model_name)
            else:
                model = YOLO(model_path)
            return model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None

    def benchmark_model(self, model, model_name: str, test_images: list, num_runs: int = 100):
        """
        Benchmark a single model on test images.

        Args:
            model: YOLO model instance
            model_name (str): Name of the model
            test_images (list): List of image paths
            num_runs (int): Number of inference runs for averaging

        Returns:
            dict: Benchmark results
        """
        print(f"\nBenchmarking {model_name}...")

        if len(test_images) == 0:
            raise ValueError("test_images list is empty")

        runs = min(num_runs, len(test_images))

        # Warm-up runs
        for _ in range(5):
            try:
                _ = model(test_images[0], verbose=False)
            except Exception:
                # Some models may expect specific input types; ignore warm-up errors
                pass

        # Timing runs
        start_time = time.time()
        for img_path in test_images[:runs]:
            _ = model(img_path, verbose=False)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_image = total_time / runs if runs > 0 else float('inf')
        fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0.0

        # Get model info (file size if a local .pt exists)
        model_file = Path(f"{model_name}.pt")
        model_size = model_file.stat().st_size / (1024 * 1024) if model_file.exists() else None

        benchmark_result = {
            'model': model_name,
            'fps': round(fps, 2),
            'latency_ms': round(avg_time_per_image * 1000, 2),
            'model_size_mb': round(model_size, 2) if model_size is not None else None,
            'device': self.device
        }

        print(f"Results: {fps:.2f} FPS, {avg_time_per_image*1000:.2f}ms latency")

        self.results.append(benchmark_result)
        return benchmark_result

    def run_benchmark(self, models: list, test_images: list, num_runs: int = 100):
        """
        Run benchmark on multiple models.

        Args:
            models (list): List of model names to benchmark
            test_images (list): List of test image paths
            num_runs (int): Number of runs per model
        """
        print(f"Starting benchmark on {len(models)} models...")

        for model_name in models:
            model = self.load_model(model_name)
            if model is not None:
                try:
                    self.benchmark_model(model, model_name, test_images, num_runs=num_runs)
                except Exception as e:
                    print(f"Benchmark failed for {model_name}: {e}")

        return self.get_results()

    def get_results(self):
        """Return benchmark results as DataFrame."""
        return pd.DataFrame(self.results)

    def save_results(self, output_path: str = 'results/benchmark_results.csv'):
        """Save results to CSV file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = self.get_results()
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        return df


def collect_sample_images(sample_dir: str = 'data/samples', num: int = 100):
    """Collect up to `num` images from `sample_dir` (recursively)."""
    p = Path(sample_dir)
    if not p.exists():
        return []
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    imgs = [str(x) for x in p.rglob('*') if x.suffix.lower() in exts]
    return imgs[:num]


def main():
    """Main function to run benchmark from command line."""
    parser = argparse.ArgumentParser(description='YOLO Speed Benchmark')
    parser.add_argument('--models', nargs='+',
                        default=['yolov5n', 'yolov8n', 'yolov11n'],
                        help='Models to benchmark')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of sample images to test')
    parser.add_argument('--output', type=str,
                        default='results/benchmark_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--sample-dir', type=str, default='data/samples',
                        help='Directory with sample images to benchmark')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of inference runs per model')

    args = parser.parse_args()

    # Gather test images from sample directory if available
    test_images = collect_sample_images(args.sample_dir, args.samples)
    if not test_images:
        print(f"No sample images found in '{args.sample_dir}'. Using placeholder paths.")
        test_images = ['path/to/test/image.jpg'] * args.samples

    # Initialize and run benchmark
    benchmark = YOLOBenchmark()
    benchmark.run_benchmark(args.models, test_images, num_runs=args.runs)
    benchmark.save_results(args.output)

    # Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(benchmark.get_results().to_string(index=False))


if __name__ == '__main__':
    main()
