"""
Data preprocessing pipeline for Credit Card Fraud Detection.

This module handles:
1. Feature scaling (StandardScaler for Amount and Time)
2. Class imbalance handling (SMOTE)
3. Train/validation/test splitting with stratification
4. Pipeline orchestration for reproducible preprocessing

Author: Your Name
Date: 2024
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
import joblib
import logging
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    smote_sampling_strategy: float = 0.5  # Balance to 50% fraud after SMOTE
    smote_k_neighbors: int = 5
    scale_features: Tuple[str, ...] = ('Amount', 'Time')
    target_column: str = 'Class'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        return {
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'smote_sampling_strategy': self.smote_sampling_strategy,
            'smote_k_neighbors': self.smote_k_neighbors,
            'scale_features': list(self.scale_features),
            'target_column': self.target_column
        }


class DataValidator:
    """Validate data quality before preprocessing"""
    
    @staticmethod
    def validate_input(df: pd.DataFrame, config: PreprocessingConfig) -> None:
        """
        Validate input dataframe meets requirements.
        
        Args:
            df: Input dataframe
            config: Preprocessing configuration
            
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating input data...")
        
        # Check required columns
        required_cols = list(config.scale_features) + [config.target_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0].index.tolist()
            raise ValueError(f"Found missing values in columns: {null_cols}")
        
        # Check target column values
        unique_targets = df[config.target_column].unique()
        if not all(t in [0, 1] for t in unique_targets):
            raise ValueError(f"Target column must contain only 0 and 1. Found: {unique_targets}")
        
        # Check class imbalance warning
        fraud_rate = df[config.target_column].mean()
        if fraud_rate < 0.01:
            logger.warning(f"Extreme class imbalance detected: {fraud_rate:.4%} fraud rate")
            logger.warning("SMOTE will be applied to handle imbalance")
        
        logger.info("✅ Data validation passed")
    
    @staticmethod
    def validate_post_split(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Validate data splits are correct.
        
        Args:
            X_train, X_val, X_test: Feature splits
            y_train, y_val, y_test: Target splits
        """
        # Check no leakage between splits
        train_indices = set(X_train.index) if hasattr(X_train, 'index') else set(range(len(X_train)))
        val_indices = set(X_val.index) if hasattr(X_val, 'index') else set(range(len(X_val)))
        test_indices = set(X_test.index) if hasattr(X_test, 'index') else set(range(len(X_test)))
        
        assert len(train_indices & val_indices) == 0, "Train and val indices overlap!"
        assert len(train_indices & test_indices) == 0, "Train and test indices overlap!"
        assert len(val_indices & test_indices) == 0, "Val and test indices overlap!"
        
        # Check class distribution
        logger.info(f"Train fraud rate: {y_train.mean():.4%}")
        logger.info(f"Validation fraud rate: {y_val.mean():.4%}")
        logger.info(f"Test fraud rate: {y_test.mean():.4%}")
        
        logger.info("✅ Post-split validation passed")


class FeatureScaler:
    """Scale specific features using StandardScaler"""
    
    def __init__(self, features_to_scale: list):
        """
        Initialize scaler.
        
        Args:
            features_to_scale: List of feature names to scale
        """
        self.features_to_scale = features_to_scale
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit scaler on training data.
        
        Args:
            X: Training features
            
        Returns:
            self: Fitted scaler
        """
        logger.info(f"Fitting scaler on features: {self.features_to_scale}")
        self.scaler.fit(X[self.features_to_scale])
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_scaled = X.copy()
        X_scaled[self.features_to_scale] = self.scaler.transform(X[self.features_to_scale])
        
        logger.debug(f"Scaled {len(self.features_to_scale)} features")
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def save(self, path: str) -> None:
        """Save scaler to disk"""
        joblib.dump(self, path)
        logger.info(f"Scaler saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'FeatureScaler':
        """Load scaler from disk"""
        return joblib.load(path)


class DataPreprocessor:
    """
    Main preprocessing pipeline orchestrator.
    
    Handles:
    1. Train/validation/test splitting
    2. Feature scaling
    3. SMOTE for class imbalance
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or PreprocessingConfig()
        self.scaler = None
        self.smote = None
        self.is_fitted = False
        
        logger.info(f"Initialized DataPreprocessor with config: {self.config.to_dict()}")
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                    pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Splitting data into train/val/test...")
        
        # Separate features and target
        X = df.drop(self.config.target_column, axis=1)
        y = df[self.config.target_column]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y  # Preserve class distribution
        )
        
        # Second split: separate validation from remaining
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train fraud rate: {y_train.mean():.4%}")
        logger.info(f"Val fraud rate: {y_val.mean():.4%}")
        logger.info(f"Test fraud rate: {y_test.mean():.4%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_preprocessing_pipeline(self) -> ImbPipeline:
        """
        Create a scikit-learn pipeline for preprocessing.
        
        Returns:
            Pipeline with scaling and SMOTE
        """
        logger.info("Creating preprocessing pipeline...")
        
        # Create feature scaler
        self.scaler = FeatureScaler(list(self.config.scale_features))
        
        # Create SMOTE for handling imbalance
        self.smote = SMOTE(
            sampling_strategy=self.config.smote_sampling_strategy,
            k_neighbors=self.config.smote_k_neighbors,
            random_state=self.config.random_state
        )
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('scaler', self.scaler),
            ('smote', self.smote)
        ])
        
        logger.info("✅ Preprocessing pipeline created")
        return pipeline
    
    def fit_preprocessor(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'DataPreprocessor':
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            self: Fitted preprocessor
        """
        logger.info("Fitting preprocessing pipeline...")
        
        pipeline = self.create_preprocessing_pipeline()
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Store fitted components
        self.scaler = pipeline.named_steps['scaler']
        self.smote = pipeline.named_steps['smote']
        self.is_fitted = True
        
        logger.info("✅ Preprocessing pipeline fitted")
        return self
    
    def transform_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform
            y: Optional targets (for SMOTE)
            
        Returns:
            Transformed features and optionally transformed targets
        """
        if not self.is_fitted:
            raise ValueError("Must fit preprocessor before transform")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply SMOTE if targets provided
        if y is not None:
            X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
            logger.info(f"Applied SMOTE: {len(y)} -> {len(y_resampled)} samples")
            logger.info(f"New fraud rate: {y_resampled.mean():.4%}")
            return X_resampled, y_resampled
        
        return X_scaled, None
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Transformed features and targets
        """
        self.fit_preprocessor(X_train, y_train)
        return self.transform_features(X_train, y_train)
    
    def save(self, path: str) -> None:
        """
        Save preprocessor to disk.
        
        Args:
            path: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        save_dict = {
            'config': self.config.to_dict(),
            'scaler': self.scaler,
            'smote': self.smote,
            'is_fitted': self.is_fitted
        }
        joblib.dump(save_dict, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            path: Path to saved preprocessor
            
        Returns:
            Loaded preprocessor
        """
        save_dict = joblib.load(path)
        
        # Reconstruct preprocessor
        config = PreprocessingConfig(**{k: v for k, v in save_dict['config'].items()})
        preprocessor = DataPreprocessor(config)
        preprocessor.scaler = save_dict['scaler']
        preprocessor.smote = save_dict['smote']
        preprocessor.is_fitted = save_dict['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


class PreprocessingReporter:
    """Generate reports on preprocessing steps"""
    
    @staticmethod
    def generate_report(original_df: pd.DataFrame, 
                       processed_shapes: Dict[str, Tuple[int, int]],
                       config: PreprocessingConfig) -> Dict[str, Any]:
        """
        Generate comprehensive preprocessing report.
        
        Args:
            original_df: Original dataframe
            processed_shapes: Dictionary with shapes after preprocessing
            config: Preprocessing configuration
            
        Returns:
            Dictionary with report data
        """
        original_fraud_rate = original_df[config.target_column].mean()
        
        report = {
            'original_data': {
                'n_samples': len(original_df),
                'n_features': original_df.shape[1] - 1,  # Exclude target
                'fraud_rate': float(original_fraud_rate),
                'n_fraud': int(original_df[config.target_column].sum()),
                'n_legit': int(len(original_df) - original_df[config.target_column].sum())
            },
            'processed_data': {
                k: {'n_samples': v[0], 'n_features': v[1]} 
                for k, v in processed_shapes.items()
            },
            'config': config.to_dict(),
            'preprocessing_steps': [
                'Split data into train/val/test (stratified)',
                f'Scaled features: {list(config.scale_features)}',
                f'Applied SMOTE with sampling_strategy={config.smote_sampling_strategy}'
            ]
        }
        
        # Save report
        with open('data/preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Preprocessing report saved to data/preprocessing_report.json")
        return report


# Convenience function for simple use cases
def create_preprocessed_data(df: pd.DataFrame, 
                            config: Optional[PreprocessingConfig] = None,
                            save_preprocessor: bool = True) -> Tuple[DataPreprocessor, Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    """
    One-stop function to preprocess data for modeling.
    
    Args:
        df: Input dataframe
        config: Preprocessing configuration
        save_preprocessor: Whether to save the fitted preprocessor
        
    Returns:
        Tuple of (preprocessor, X_dict, y_dict)
        where X_dict and y_dict contain 'train', 'val', 'test' keys
    """
    # Initialize
    preprocessor = DataPreprocessor(config)
    validator = DataValidator()
    
    # Validate input
    validator.validate_input(df, preprocessor.config)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
    
    # Fit and transform training data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform validation and test data (no SMOTE)
    X_val_processed, _ = preprocessor.transform_features(X_val, y_val)
    X_test_processed, _ = preprocessor.transform_features(X_test, y_test)
    
    # Validate splits
    validator.validate_post_split(
        X_train_processed, y_train_processed,
        X_val_processed, y_val,
        X_test_processed, y_test
    )
    
    # Package results
    X_dict = {
        'train': X_train_processed,
        'val': X_val_processed,
        'test': X_test_processed
    }
    
    y_dict = {
        'train': y_train_processed,
        'val': y_val,
        'test': y_test
    }
    
    # Generate report
    processed_shapes = {
        'train': X_train_processed.shape,
        'val': X_val_processed.shape,
        'test': X_test_processed.shape
    }
    PreprocessingReporter.generate_report(df, processed_shapes, preprocessor.config)
    
    # Save preprocessor if requested
    if save_preprocessor:
        preprocessor.save('models/preprocessor.pkl')
    
    logger.info("✅ Preprocessing complete!")
    logger.info(f"Training set after SMOTE: {X_train_processed.shape[0]} samples")
    logger.info(f"Training fraud rate after SMOTE: {y_train_processed.mean():.4%}")
    
    return preprocessor, X_dict, y_dict


if __name__ == "__main__":
    # Example usage and testing
    print("="*60)
    print("Testing Data Preprocessing Module")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/creditcard.csv')
    print(f"Loaded {len(df)} transactions")
    
    # Run preprocessing
    preprocessor, X_dict, y_dict = create_preprocessed_data(df)
    
    print("\n✅ Preprocessing successful!")
    print(f"Train shape: {X_dict['train'].shape}")
    print(f"Validation shape: {X_dict['val'].shape}")
    print(f"Test shape: {X_dict['test'].shape}")
    print(f"\nPreprocessor saved to: models/preprocessor.pkl")
    print(f"Report saved to: data/preprocessing_report.json")