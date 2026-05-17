"""
Script to preprocess the credit card fraud dataset.
Run this before training models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.preprocess import create_preprocessed_data, PreprocessingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main preprocessing pipeline"""
    
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'creditcard.csv'))
    print(f"✅ Loaded {len(df):,} transactions")
    
    # Configure preprocessing
    config = PreprocessingConfig(
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        smote_sampling_strategy=0.5,  # Balance to 50% fraud
        smote_k_neighbors=5
    )
    
    # Run preprocessing
    logger.info("Starting preprocessing...")
    preprocessor, X_dict, y_dict = create_preprocessed_data(
        df, 
        config=config,
        save_preprocessor=True
    )
    
    # Save processed data for later use
    logger.info("Saving processed data...")

    # Ensure output directories exist (fixes occasional crash in fresh envs)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    for split in ['train', 'val', 'test']:
        X_dict[split].to_csv(f'data/X_{split}.csv', index=False)
        y_dict[split].to_csv(f'data/y_{split}.csv', index=False)

    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Train shape: {X_dict['train'].shape}")
    print(f"Validation shape: {X_dict['val'].shape}")
    print(f"Test shape: {X_dict['test'].shape}")
    print(f"\nFiles saved:")
    print("  - models/preprocessor.pkl")
    print("  - data/preprocessing_report.json")
    print("  - data/X_train.csv, X_val.csv, X_test.csv")
    print("  - data/y_train.csv, y_val.csv, y_test.csv")


if __name__ == "__main__":
    main()