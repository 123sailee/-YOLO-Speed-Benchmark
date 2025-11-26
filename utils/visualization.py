"""Visualization utilities for benchmark results."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_fps_comparison(results: pd.DataFrame, output_path: str = 'results/fps_comparison.png'):
    """
    Create a bar plot comparing FPS across models.
    
    Args:
        results (pd.DataFrame): Benchmark results
        output_path (str): Path to save the plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(results['model'], results['fps'], color='steelblue')
    plt.xlabel('Model')
    plt.ylabel('FPS (Frames Per Second)')
    plt.title('YOLO Model FPS Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved FPS comparison plot to {output_path}")


def plot_latency_comparison(results: pd.DataFrame, output_path: str = 'results/latency_comparison.png'):
    """
    Create a bar plot comparing latency across models.
    
    Args:
        results (pd.DataFrame): Benchmark results
        output_path (str): Path to save the plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(results['model'], results['latency_ms'], color='coral')
    plt.xlabel('Model')
    plt.ylabel('Latency (ms)')
    plt.title('YOLO Model Latency Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved latency comparison plot to {output_path}")
