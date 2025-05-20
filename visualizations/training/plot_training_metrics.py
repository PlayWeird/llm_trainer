#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility for plotting training metrics from TensorBoard logs or saved CSV files.
Supports visualization of loss curves, learning rates, and other metrics.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import glob
import re


def load_tensorboard_data(log_dir):
    """
    Load training metrics from TensorBoard event files.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        
    Returns:
        A dictionary of DataFrames with metrics
    """
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    # Sort event files by timestamp to get the latest
    event_files.sort()
    latest_event_file = event_files[-1]
    
    # Load the events from the file
    ea = event_accumulator.EventAccumulator(latest_event_file)
    ea.Reload()
    
    # Extract available tags (metrics)
    tags = ea.Tags()["scalars"]
    
    # Extract data for each tag
    data = {}
    for tag in tags:
        tag_data = pd.DataFrame(ea.Scalars(tag))
        # Clean tag name for dict key
        clean_tag = re.sub(r'[/]', '_', tag)
        data[clean_tag] = tag_data
    
    return data


def load_csv_data(csv_file):
    """
    Load training metrics from a CSV file.
    
    Args:
        csv_file: Path to the CSV file containing training metrics
        
    Returns:
        A pandas DataFrame with metrics
    """
    return pd.read_csv(csv_file)


def plot_training_loss(data, output_dir=None, show_plot=True):
    """
    Plot training loss curve.
    
    Args:
        data: DataFrame or dictionary containing training loss data
        output_dir: Optional directory to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    if isinstance(data, dict):
        # For TensorBoard data
        for key, df in data.items():
            if "loss" in key.lower():
                plt.plot(df["step"], df["value"], label=key)
    else:
        # For CSV data
        loss_cols = [col for col in data.columns if "loss" in col.lower()]
        for col in loss_cols:
            plt.plot(data["step"] if "step" in data.columns else range(len(data)), 
                    data[col], label=col)
    
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Use MaxNLocator to limit the number of ticks on the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_learning_rate(data, output_dir=None, show_plot=True):
    """
    Plot learning rate curve.
    
    Args:
        data: DataFrame or dictionary containing learning rate data
        output_dir: Optional directory to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    if isinstance(data, dict):
        # For TensorBoard data
        for key, df in data.items():
            if "learning" in key.lower() and "rate" in key.lower():
                plt.plot(df["step"], df["value"], label=key)
    else:
        # For CSV data
        lr_cols = [col for col in data.columns if "learning" in col.lower() and "rate" in col.lower()]
        for col in lr_cols:
            plt.plot(data["step"] if "step" in data.columns else range(len(data)), 
                    data[col], label=col)
    
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Use MaxNLocator to limit the number of ticks on the x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_metrics_grid(data, output_dir=None, show_plot=True):
    """
    Plot a grid of all available metrics.
    
    Args:
        data: DataFrame or dictionary containing metrics data
        output_dir: Optional directory to save the plot
        show_plot: Whether to display the plot
    """
    # Set the style
    sns.set(style="whitegrid")
    
    if isinstance(data, dict):
        # For TensorBoard data
        metrics = list(data.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            print("No metrics found to plot")
            return
        
        # Calculate grid dimensions
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                df = data[metric]
                axes[i].plot(df["step"], df["value"])
                axes[i].set_title(metric)
                axes[i].set_xlabel("Step")
                axes[i].set_ylabel("Value")
                axes[i].grid(True, linestyle="--", alpha=0.7)
        
        # Hide unused subplots
        for j in range(n_metrics, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        
    else:
        # For CSV data
        metrics = [col for col in data.columns if col != "step" and col != "epoch"]
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            print("No metrics found to plot")
            return
        
        # Calculate grid dimensions
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        x = data["step"] if "step" in data.columns else range(len(data))
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].plot(x, data[metric])
                axes[i].set_title(metric)
                axes[i].set_xlabel("Step")
                axes[i].set_ylabel("Value")
                axes[i].grid(True, linestyle="--", alpha=0.7)
        
        # Hide unused subplots
        for j in range(n_metrics, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "metrics_grid.png"), dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        help="Path to TensorBoard logs directory",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        help="Path to CSV file with metrics",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Whether to display plots (in addition to saving)",
    )
    
    args = parser.parse_args()
    
    if not args.tensorboard_dir and not args.csv_file:
        parser.error("Either --tensorboard_dir or --csv_file must be provided")
    
    # Load data
    if args.tensorboard_dir:
        print(f"Loading TensorBoard data from {args.tensorboard_dir}")
        data = load_tensorboard_data(args.tensorboard_dir)
    else:
        print(f"Loading CSV data from {args.csv_file}")
        data = load_csv_data(args.csv_file)
    
    # Create plots
    print("Creating plots...")
    plot_training_loss(data, args.output_dir, args.show_plots)
    plot_learning_rate(data, args.output_dir, args.show_plots)
    plot_metrics_grid(data, args.output_dir, args.show_plots)
    
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()