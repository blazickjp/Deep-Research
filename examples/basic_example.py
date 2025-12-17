"""
Basic example of training a deep linear network on synthetic time series data.

This script demonstrates:
1. Creating synthetic time series data
2. Initializing a deep linear network
3. Training the model
4. Evaluating performance
5. Visualizing results
"""

import torch
from deep_linear_ts import DeepLinearTimeSeries
from deep_linear_ts.data import create_synthetic_dataset
from deep_linear_ts.train import train_model
from deep_linear_ts.evaluate import evaluate_model, compare_with_baselines
from deep_linear_ts.utils import get_model_summary, DynamicsAnalyzer, visualize_predictions


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    print("="*60)
    print("DEEP LINEAR NETWORKS FOR TIME SERIES - BASIC EXAMPLE")
    print("="*60)

    # Step 1: Create synthetic dataset
    print("\n[1/5] Creating synthetic dataset...")
    train_loader, val_loader, test_loader = create_synthetic_dataset(
        n_samples=10000,
        n_features=7,
        sequence_length=96,
        prediction_length=24,
        batch_size=32,
        pattern='sine'  # Try 'sine', 'linear', 'ar', or 'random'
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Step 2: Initialize model
    print("\n[2/5] Initializing deep linear network...")
    model = DeepLinearTimeSeries(
        input_dim=7,
        hidden_dim=64,
        output_dim=7,
        depth=100,  # 100 layers!
        sequence_length=96,
        prediction_length=24,  # Forecast 24 steps ahead
        temporal_mixing='toeplitz',  # or 'linear_attention', 'fourier'
        use_residual=True,
        residual_weight=0.1,
    )

    # Print model summary
    print(get_model_summary(model))

    # Step 3: Train the model
    print("\n[3/5] Training model...")

    # Initialize geometry analyzer
    analyzer = DynamicsAnalyzer()

    history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=20,
        learning_rate=1e-3,
        checkpoint_path='checkpoints/basic_example.pt',
        early_stopping_patience=5,
    )

    print(f"\nBest validation loss: {min(history['val_loss']):.4f}")

    # Step 4: Evaluate the model
    print("\n[4/5] Evaluating model...")

    # Comprehensive evaluation
    results = compare_with_baselines(model, test_loader)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric:20s}: {value:.4f}")

    # Step 5: Visualize predictions
    print("\n[5/5] Visualizing predictions...")

    visualize_predictions(
        model,
        test_loader,
        n_samples=3,
        save_path='predictions.png'
    )

    print("\n" + "="*60)
    print("EXAMPLE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Try different depths: depth=10, 50, 500, 1000")
    print("2. Experiment with temporal mixing: 'toeplitz', 'linear_attention', 'fourier'")
    print("3. Load real datasets: load_ett_dataset('ETTh1')")
    print("4. Analyze geometry: use DynamicsAnalyzer to track training")


if __name__ == '__main__':
    main()
