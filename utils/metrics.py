"""Metrics computation for YOLO benchmarks."""

import pandas as pd
from typing import List, Dict


def compute_summary_stats(results: pd.DataFrame) -> Dict:
    """
    Compute summary statistics from benchmark results.
    
    Args:
        results (pd.DataFrame): Benchmark results DataFrame
    
    Returns:
        dict: Summary statistics
    """
    return {
        'total_models': len(results),
        'avg_fps': results['fps'].mean(),
        'avg_latency_ms': results['latency_ms'].mean(),
        'fastest_model': results.loc[results['fps'].idxmax(), 'model'],
        'slowest_model': results.loc[results['fps'].idxmin(), 'model'],
    }


def rank_models_by_fps(results: pd.DataFrame) -> pd.DataFrame:
    """Rank models by FPS (highest first)."""
    return results.sort_values('fps', ascending=False).reset_index(drop=True)


def rank_models_by_latency(results: pd.DataFrame) -> pd.DataFrame:
    """Rank models by latency (lowest first)."""
    return results.sort_values('latency_ms', ascending=True).reset_index(drop=True)
