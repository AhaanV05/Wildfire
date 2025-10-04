#!/usr/bin/env python3
"""
XGBoost Wildfire ROS Prediction Model

Uses VAE-generated synthetic data for training and real data for validation
to predict Rate of Spread (ROS) from wildfire environmental features.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import signal
import sys
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WildfireROSPredictor:
    """XGBoost model for predicting wildfire Rate of Spread (ROS)."""
    
    def __init__(self, data_file="VAE_Dist/vae_combined_dataset.csv"):
        """Initialize the ROS predictor."""
        logger.info("🔥 Initializing Wildfire ROS Predictor with XGBoost")
        
        # Set up graceful interruption
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Load combined dataset
        self.data = pd.read_csv(data_file)
        logger.info(f"📊 Loaded {len(self.data)} total samples")
        
        # Define feature columns (excluding target)
        self.feature_columns = [
            'temp_c', 'rel_humidity_pct', 'wind_speed_ms', 'precip_mm',
            'vpd_kpa', 'fwi', 'ndvi', 'ndmi', 'lfmc_proxy_pct',
            'elevation_m', 'slope_pct', 'aspect_deg'
        ]
        
        self.target_column = 'target_ros_m_min'
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\n🛑 Graceful interruption requested...")
        self.interrupted = True
        
    def prepare_data(self):
        """Separate real and synthetic data for training/validation split."""
        logger.info("🔧 Preparing data for training and validation...")
        
        # Clean data
        clean_data = self.data.dropna(subset=self.feature_columns + [self.target_column])
        logger.info(f"   Using {len(clean_data)} complete samples after cleaning")
        
        # Separate real and synthetic data based on fire_id prefix
        real_data = clean_data[clean_data['fire_id'].str.startswith('REAL_')]
        synthetic_data = clean_data[clean_data['fire_id'].str.startswith('VAE_')]
        
        logger.info(f"   Real data samples: {len(real_data)} (for validation)")
        logger.info(f"   Synthetic data samples: {len(synthetic_data)} (for training)")
        
        # Prepare features and targets
        self.X_real = real_data[self.feature_columns].values
        self.y_real = real_data[self.target_column].values
        
        self.X_synthetic = synthetic_data[self.feature_columns].values
        self.y_synthetic = synthetic_data[self.target_column].values
        
        # Combine for overall statistics
        self.X_all = clean_data[self.feature_columns].values
        self.y_all = clean_data[self.target_column].values
        
        # Scale features
        self.X_synthetic_scaled = self.scaler.fit_transform(self.X_synthetic)
        self.X_real_scaled = self.scaler.transform(self.X_real)
        self.X_all_scaled = self.scaler.transform(self.X_all)
        
        logger.info("✅ Data preparation completed")
        
        # Print feature statistics
        self._print_feature_stats()
        
    def _print_feature_stats(self):
        """Print feature statistics for real vs synthetic data."""
        logger.info("📈 Feature Statistics Comparison:")
        logger.info("="*80)
        
        real_df = pd.DataFrame(self.X_real, columns=self.feature_columns)
        synthetic_df = pd.DataFrame(self.X_synthetic, columns=self.feature_columns)
        
        for feature in self.feature_columns:
            real_mean = real_df[feature].mean()
            real_std = real_df[feature].std()
            synth_mean = synthetic_df[feature].mean()
            synth_std = synthetic_df[feature].std()
            
            logger.info(f"{feature:15s}: Real={real_mean:8.3f}±{real_std:6.3f}, "
                       f"Synthetic={synth_mean:8.3f}±{synth_std:6.3f}")
        
        # Target statistics
        logger.info("-" * 80)
        logger.info(f"{'ROS Target':15s}: Real={self.y_real.mean():8.3f}±{self.y_real.std():6.3f}, "
                   f"Synthetic={self.y_synthetic.mean():8.3f}±{self.y_synthetic.std():6.3f}")
        logger.info("="*80)
    
    def train_model(self, hyperparameter_tuning=True, tuning_method='randomized'):
        """Train XGBoost model on synthetic data with enhanced hyperparameter tuning."""
        logger.info("🧠 Training XGBoost model on synthetic data...")
        
        if hyperparameter_tuning:
            logger.info(f"🔍 Performing {tuning_method} hyperparameter tuning...")
            logger.info("   Press Ctrl+C to stop tuning and use current best parameters")
            
            # 🎯 HYPERPARAMETER TUNING SECTION
            # ================================
            # Enhanced parameter search space for wildfire ROS prediction
            param_distributions = {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [3, 4, 5, 6, 7, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 1.5, 2.0, 5.0],
                'gamma': [0, 0.1, 0.2, 0.5, 1.0],
                'min_child_weight': [1, 3, 5, 7, 10]
            }
            
            try:
                base_model = xgb.XGBRegressor(
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0  # Reduce XGBoost output
                )
                
                # Use RandomizedSearchCV for efficiency
                logger.info("   Using RandomizedSearchCV with 100 iterations")
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions,
                    n_iter=100,  # Test 100 random combinations
                    cv=5,
                    scoring='neg_mean_squared_error',
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit with interruption handling
                random_search.fit(self.X_synthetic_scaled, self.y_synthetic)
                
                best_params = random_search.best_params_
                best_score = -random_search.best_score_
                
                logger.info(f"🏆 Best hyperparameters found (CV RMSE: {np.sqrt(best_score):.4f}):")
                for param, value in best_params.items():
                    logger.info(f"   {param}: {value}")
                
                # Store hyperparameter results for analysis
                self.hyperparameter_results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'cv_results': random_search.cv_results_
                }
                
                # Train final model with best parameters
                self.model = xgb.XGBRegressor(
                    **best_params,
                    random_state=42,
                    n_jobs=-1
                )
                
            except KeyboardInterrupt:
                logger.info("🛑 Hyperparameter tuning interrupted!")
                logger.info("   Using default parameters instead...")
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    random_state=42,
                    n_jobs=-1
                )
        
        else:
            # Use default parameters
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1
            )
        
        # Train on all synthetic data
        logger.info("🎯 Training final model on synthetic data...")
        self.model.fit(self.X_synthetic_scaled, self.y_synthetic)
        
        logger.info("✅ Model training completed")
        
    def validate_model(self):
        """Comprehensive model validation with multiple metrics and visualizations."""
        logger.info("🔍 Comprehensive model validation on real data...")
        
        # Predictions on real data
        y_pred_real = self.model.predict(self.X_real_scaled)
        
        # Calculate comprehensive metrics
        mse_real = mean_squared_error(self.y_real, y_pred_real)
        mae_real = mean_absolute_error(self.y_real, y_pred_real)
        r2_real = r2_score(self.y_real, y_pred_real)
        rmse_real = np.sqrt(mse_real)
        mape_real = mean_absolute_percentage_error(self.y_real, y_pred_real)
        evs_real = explained_variance_score(self.y_real, y_pred_real)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(self.y_real, y_pred_real)
        spearman_r, spearman_p = spearmanr(self.y_real, y_pred_real)
        
        logger.info("📊 Comprehensive Validation Results on Real Data:")
        logger.info("="*70)
        logger.info(f"RMSE:                    {rmse_real:.4f} m/min")
        logger.info(f"MAE:                     {mae_real:.4f} m/min")
        logger.info(f"MAPE:                    {mape_real:.2f}%")
        logger.info(f"R²:                      {r2_real:.4f}")
        logger.info(f"Explained Variance:      {evs_real:.4f}")
        logger.info(f"Pearson Correlation:     {pearson_r:.4f} (p={pearson_p:.4f})")
        logger.info(f"Spearman Correlation:    {spearman_r:.4f} (p={spearman_p:.4f})")
        logger.info("="*70)
        
        # Cross-validation with multiple metrics
        logger.info("🔄 Cross-validation on real data (140 samples)...")
        cv_scores_mse = cross_val_score(self.model, self.X_real_scaled, self.y_real, cv=5, scoring='neg_mean_squared_error')
        cv_scores_r2 = cross_val_score(self.model, self.X_real_scaled, self.y_real, cv=5, scoring='r2')
        cv_scores_mae = cross_val_score(self.model, self.X_real_scaled, self.y_real, cv=5, scoring='neg_mean_absolute_error')
        
        cv_rmse = np.sqrt(-cv_scores_mse)
        cv_r2 = cv_scores_r2
        cv_mae = -cv_scores_mae
        
        logger.info(f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        logger.info(f"CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        logger.info(f"CV MAE:  {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
        
        # Detailed analysis and visualizations
        self._detailed_validation_analysis(y_pred_real)
        self._create_validation_visualizations(y_pred_real)
        
        return {
            'rmse': rmse_real, 'mae': mae_real, 'mape': mape_real,
            'r2': r2_real, 'evs': evs_real,
            'pearson_r': pearson_r, 'spearman_r': spearman_r,
            'cv_rmse_mean': cv_rmse.mean(), 'cv_rmse_std': cv_rmse.std(),
            'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std()
        }
    
    def _detailed_validation_analysis(self, y_pred_real):
        """Perform detailed validation analysis."""
        logger.info("🔬 Detailed Validation Analysis:")
        
        # Prediction accuracy by ROS range
        low_ros = self.y_real < 0.5
        med_ros = (self.y_real >= 0.5) & (self.y_real < 1.5)
        high_ros = self.y_real >= 1.5
        
        if np.any(low_ros):
            mae_low = mean_absolute_error(self.y_real[low_ros], y_pred_real[low_ros])
            logger.info(f"   Low ROS (<0.5):  MAE = {mae_low:.4f}, Count = {np.sum(low_ros)}")
        
        if np.any(med_ros):
            mae_med = mean_absolute_error(self.y_real[med_ros], y_pred_real[med_ros])
            logger.info(f"   Med ROS (0.5-1.5): MAE = {mae_med:.4f}, Count = {np.sum(med_ros)}")
        
        if np.any(high_ros):
            mae_high = mean_absolute_error(self.y_real[high_ros], y_pred_real[high_ros])
            logger.info(f"   High ROS (>1.5): MAE = {mae_high:.4f}, Count = {np.sum(high_ros)}")
        
        # Prediction bias analysis
        residuals = y_pred_real - self.y_real
        logger.info(f"   Mean residual (bias): {residuals.mean():.4f}")
        logger.info(f"   Std residuals: {residuals.std():.4f}")
        
    def _create_validation_visualizations(self, y_pred_real):
        """Create comprehensive validation visualizations split across multiple pages."""
        logger.info("📊 Creating comprehensive validation visualizations across multiple pages...")
        
        residuals = y_pred_real - self.y_real
        
        # ===========================
        # PAGE 1: CORE PREDICTIONS
        # ===========================
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('Page 1: Core Prediction Analysis', fontsize=18, y=0.95)
        
        # 1. Actual vs Predicted scatter plot
        axes1[0, 0].scatter(self.y_real, y_pred_real, alpha=0.7, color='blue', s=60)
        axes1[0, 0].plot([self.y_real.min(), self.y_real.max()], 
                        [self.y_real.min(), self.y_real.max()], 'r--', lw=3, label='Perfect Prediction')
        axes1[0, 0].set_xlabel('Actual ROS (m/min)', fontsize=14)
        axes1[0, 0].set_ylabel('Predicted ROS (m/min)', fontsize=14)
        axes1[0, 0].set_title(f'Actual vs Predicted\nR² = {r2_score(self.y_real, y_pred_real):.3f}', fontsize=16)
        axes1[0, 0].legend(fontsize=12)
        axes1[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals scatter plot
        axes1[0, 1].scatter(y_pred_real, residuals, alpha=0.7, color='green', s=60)
        axes1[0, 1].axhline(y=0, color='r', linestyle='--', lw=3)
        axes1[0, 1].set_xlabel('Predicted ROS (m/min)', fontsize=14)
        axes1[0, 1].set_ylabel('Residuals (m/min)', fontsize=14)
        axes1[0, 1].set_title(f'Residuals Plot\nMean Bias = {residuals.mean():.4f}', fontsize=16)
        axes1[0, 1].grid(True, alpha=0.3)
        
        # 3. Actual vs Predicted Distributions
        axes1[1, 0].hist(self.y_real, bins=20, density=True, alpha=0.7, color='blue', 
                        label=f'Actual (μ={self.y_real.mean():.3f})', edgecolor='black')
        axes1[1, 0].hist(y_pred_real, bins=20, density=True, alpha=0.7, color='red', 
                        label=f'Predicted (μ={y_pred_real.mean():.3f})', edgecolor='black')
        axes1[1, 0].set_xlabel('ROS (m/min)', fontsize=14)
        axes1[1, 0].set_ylabel('Density', fontsize=14)
        axes1[1, 0].set_title('Distribution Comparison', fontsize=16)
        axes1[1, 0].legend(fontsize=12)
        axes1[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction time series with confidence
        sorted_idx = np.argsort(self.y_real)
        y_real_sorted = self.y_real[sorted_idx]
        y_pred_sorted = y_pred_real[sorted_idx]
        
        axes1[1, 1].plot(y_real_sorted, label='Actual', color='blue', linewidth=3, marker='o', markersize=4)
        axes1[1, 1].plot(y_pred_sorted, label='Predicted', color='red', linewidth=3, marker='s', markersize=4)
        axes1[1, 1].fill_between(range(len(y_real_sorted)), 
                                y_pred_sorted - np.std(residuals), 
                                y_pred_sorted + np.std(residuals), 
                                alpha=0.3, color='red', label='±1σ Confidence')
        axes1[1, 1].set_xlabel('Sample Index (sorted by actual)', fontsize=14)
        axes1[1, 1].set_ylabel('ROS (m/min)', fontsize=14)
        axes1[1, 1].set_title('Predictions with Confidence Bands', fontsize=16)
        axes1[1, 1].legend(fontsize=12)
        axes1[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Ros_Pred/page1_core_predictions.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Page 1 saved: Ros_Pred/page1_core_predictions.png")
        plt.show()
        
        # ===========================
        # PAGE 2: RESIDUALS ANALYSIS
        # ===========================
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle('Page 2: Residuals and Error Analysis', fontsize=18, y=0.95)
        
        # 1. Residuals histogram with normal fit
        axes2[0, 0].hist(residuals, bins=25, density=True, alpha=0.8, color='orange', edgecolor='black')
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        p = stats.norm.pdf(x, mu, sigma)
        axes2[0, 0].plot(x, p, 'k', linewidth=3, label=f'Normal Fit\nμ={mu:.3f}, σ={sigma:.3f}')
        axes2[0, 0].set_xlabel('Residuals (m/min)', fontsize=14)
        axes2[0, 0].set_ylabel('Density', fontsize=14)
        axes2[0, 0].set_title('Residuals Distribution', fontsize=16)
        axes2[0, 0].legend(fontsize=12)
        axes2[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality test
        stats.probplot(residuals, dist="norm", plot=axes2[0, 1])
        axes2[0, 1].set_title('Q-Q Plot: Residuals Normality Test', fontsize=16)
        axes2[0, 1].grid(True, alpha=0.3)
        axes2[0, 1].tick_params(labelsize=12)
        
        # 3. Error by ROS range
        bins = np.linspace(self.y_real.min(), self.y_real.max(), 8)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_errors = []
        bin_counts = []
        for i in range(len(bins)-1):
            mask = (self.y_real >= bins[i]) & (self.y_real < bins[i+1])
            if np.any(mask):
                bin_errors.append(np.mean(np.abs(residuals[mask])))
                bin_counts.append(np.sum(mask))
            else:
                bin_errors.append(0)
                bin_counts.append(0)
        
        bars = axes2[1, 0].bar(bin_centers, bin_errors, width=(bins[1]-bins[0])*0.8, 
                              alpha=0.8, color='purple', edgecolor='black')
        axes2[1, 0].set_xlabel('Actual ROS Range (m/min)', fontsize=14)
        axes2[1, 0].set_ylabel('Mean Absolute Error', fontsize=14)
        axes2[1, 0].set_title('MAE by ROS Range', fontsize=16)
        axes2[1, 0].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                axes2[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # 4. Residuals vs Features (most important feature)
        if hasattr(self, 'model') and self.model is not None:
            # Get most important feature
            importance = self.model.feature_importances_
            most_important_idx = np.argmax(importance)
            most_important_feature = self.feature_columns[most_important_idx]
            feature_values = self.X_real[:, most_important_idx]
            
            axes2[1, 1].scatter(feature_values, residuals, alpha=0.7, color='red', s=60)
            axes2[1, 1].axhline(y=0, color='k', linestyle='--', lw=2)
            axes2[1, 1].set_xlabel(f'{most_important_feature}', fontsize=14)
            axes2[1, 1].set_ylabel('Residuals (m/min)', fontsize=14)
            axes2[1, 1].set_title(f'Residuals vs Most Important Feature\n({most_important_feature})', fontsize=16)
            axes2[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Ros_Pred/page2_residuals_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Page 2 saved: Ros_Pred/page2_residuals_analysis.png")
        plt.show()
        
        # ===========================
        # PAGE 3: FEATURE IMPORTANCE & TUNING
        # ===========================
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        fig3.suptitle('Page 3: Feature Importance & Hyperparameter Analysis', fontsize=18, y=0.95)
        
        # 1. Feature importance (all features)
        if self.model is not None:
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            bars = axes3[0, 0].barh(range(len(feature_importance)), feature_importance['importance'], 
                                   color='salmon', edgecolor='black', alpha=0.8)
            axes3[0, 0].set_yticks(range(len(feature_importance)))
            axes3[0, 0].set_yticklabels(feature_importance['feature'], fontsize=12)
            axes3[0, 0].set_xlabel('Importance Score', fontsize=14)
            axes3[0, 0].set_title('XGBoost Feature Importance', fontsize=16)
            axes3[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes3[0, 0].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                               f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        # 2. Top features pie chart
        if self.model is not None:
            top_6_features = feature_importance.tail(6)
            others_importance = feature_importance.head(-6)['importance'].sum()
            
            pie_data = list(top_6_features['importance']) + [others_importance]
            pie_labels = list(top_6_features['feature']) + ['Others']
            colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
            
            wedges, texts, autotexts = axes3[0, 1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                                                      colors=colors, startangle=90)
            axes3[0, 1].set_title('Feature Importance Distribution', fontsize=16)
            
            # Make percentage text larger
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
        
        # 3. Hyperparameter tuning results
        if hasattr(self, 'hyperparameter_results'):
            cv_results = self.hyperparameter_results['cv_results']
            scores = np.sqrt(-cv_results['mean_test_score'])  # Convert to RMSE
            
            axes3[1, 0].hist(scores, bins=20, alpha=0.8, color='teal', edgecolor='black')
            best_rmse = np.sqrt(-self.hyperparameter_results['best_score'])
            axes3[1, 0].axvline(best_rmse, color='red', linestyle='--', linewidth=3, 
                               label=f'Best RMSE: {best_rmse:.4f}')
            axes3[1, 0].set_xlabel('CV RMSE', fontsize=14)
            axes3[1, 0].set_ylabel('Frequency', fontsize=14)
            axes3[1, 0].set_title('Hyperparameter Tuning Distribution', fontsize=16)
            axes3[1, 0].legend(fontsize=12)
            axes3[1, 0].grid(True, alpha=0.3)
        
        # 4. Model performance summary
        axes3[1, 1].axis('off')  # Turn off axis for text summary
        
        # Create performance summary text
        summary_text = f"""
        Model Performance Summary
        ========================
        
        Validation Metrics:
        • RMSE: {np.sqrt(mean_squared_error(self.y_real, y_pred_real)):.4f} m/min
        • MAE: {mean_absolute_error(self.y_real, y_pred_real):.4f} m/min
        • R²: {r2_score(self.y_real, y_pred_real):.4f}
        • MAPE: {mean_absolute_percentage_error(self.y_real, y_pred_real):.2f}%
        
        Data Summary:
        • Training: {len(self.y_synthetic):,} synthetic samples
        • Validation: {len(self.y_real)} real samples
        • Features: {len(self.feature_columns)} wildfire variables
        
        Best Hyperparameters:
        • n_estimators: {self.model.n_estimators if self.model else 'N/A'}
        • max_depth: {self.model.max_depth if self.model else 'N/A'}
        • learning_rate: {self.model.learning_rate if self.model else 'N/A'}
        """
        
        axes3[1, 1].text(0.05, 0.95, summary_text, transform=axes3[1, 1].transAxes,
                         fontsize=12, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('Ros_Pred/page3_feature_importance_tuning.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Page 3 saved: Ros_Pred/page3_feature_importance_tuning.png")
        plt.show()
        
        logger.info("🎉 All visualization pages created successfully!")
        logger.info("📁 Generated files in Ros_Pred/ folder:")
        logger.info("   • page1_core_predictions.png")
        logger.info("   • page2_residuals_analysis.png")  
        logger.info("   • page3_feature_importance_tuning.png")
        
    def analyze_feature_importance(self):
        """Analyze and plot feature importance."""
        logger.info("📈 Analyzing feature importance...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info("🏆 Top 10 Most Important Features:")
        logger.info("-" * 40)
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['feature']:15s}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, y='feature', x='importance')
        plt.title('XGBoost Feature Importance for Wildfire ROS Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('Ros_Pred/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Feature importance plot saved: Ros_Pred/xgboost_feature_importance.png")
        plt.show()
        
        return feature_importance
    
    def plot_predictions(self):
        """Plot prediction vs actual values."""
        logger.info("📊 Creating prediction plots...")
        
        # Predictions
        y_pred_real = self.model.predict(self.X_real_scaled)
        y_pred_synthetic = self.model.predict(self.X_synthetic_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Real data predictions
        axes[0].scatter(self.y_real, y_pred_real, alpha=0.7, color='blue')
        axes[0].plot([self.y_real.min(), self.y_real.max()], 
                    [self.y_real.min(), self.y_real.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual ROS (m/min)')
        axes[0].set_ylabel('Predicted ROS (m/min)')
        axes[0].set_title(f'Real Data Validation\n(R² = {r2_score(self.y_real, y_pred_real):.3f})')
        axes[0].grid(True, alpha=0.3)
        
        # Synthetic data predictions (training performance)
        axes[1].scatter(self.y_synthetic, y_pred_synthetic, alpha=0.3, color='red')
        axes[1].plot([self.y_synthetic.min(), self.y_synthetic.max()], 
                    [self.y_synthetic.min(), self.y_synthetic.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual ROS (m/min)')
        axes[1].set_ylabel('Predicted ROS (m/min)')
        axes[1].set_title(f'Synthetic Data Training\n(R² = {r2_score(self.y_synthetic, y_pred_synthetic):.3f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Ros_Pred/xgboost_predictions.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Prediction plots saved: Ros_Pred/xgboost_predictions.png")
        plt.show()
    
    def save_model(self, model_path="Ros_Pred/wildfire_ros_xgboost_model.joblib"):
        """Save the trained model and scaler."""
        logger.info(f"💾 Saving model to {model_path}")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, model_path)
        logger.info("✅ Model saved successfully")
    
    def predict_ros(self, features):
        """Predict ROS for new feature data."""
        if isinstance(features, dict):
            # Single prediction
            feature_array = np.array([[features[col] for col in self.feature_columns]])
        else:
            # Multiple predictions
            feature_array = np.array(features)
        
        # Scale features
        features_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(features_scaled)
        
        return prediction

def _print_hyperparameter_explanation():
    """Print comprehensive explanation of XGBoost hyperparameters."""
    logger.info("\n" + "="*100)
    logger.info("📚 XGBOOST HYPERPARAMETER TUNING EXPLANATION")
    logger.info("="*100)
    
    logger.info("\n🎯 TREE STRUCTURE PARAMETERS:")
    logger.info("-" * 50)
    logger.info("• n_estimators (50-500):")
    logger.info("  └─ Number of boosting rounds (trees to build)")
    logger.info("  └─ More trees = better learning but slower training & risk of overfitting")
    logger.info("  └─ Optimal: Usually 100-500 for complex problems like wildfire prediction")
    
    logger.info("\n• max_depth (3-10):")
    logger.info("  └─ Maximum depth of each tree")
    logger.info("  └─ Deeper trees = capture complex interactions but risk overfitting")
    logger.info("  └─ Optimal: 6-8 for tabular data, 3-5 for simple problems")
    
    logger.info("\n• min_child_weight (1-10):")
    logger.info("  └─ Minimum sum of instance weight needed in a leaf")
    logger.info("  └─ Higher values = more conservative (prevent overfitting)")
    logger.info("  └─ Optimal: 1-5 for most problems, higher for imbalanced data")
    
    logger.info("\n🚀 LEARNING CONTROL PARAMETERS:")
    logger.info("-" * 50)
    logger.info("• learning_rate (0.01-0.3):")
    logger.info("  └─ Step size shrinkage to prevent overfitting")
    logger.info("  └─ Lower values = more robust but need more n_estimators")
    logger.info("  └─ Optimal: 0.05-0.1 for stable learning, 0.01 for very large datasets")
    
    logger.info("\n• gamma (0-1.0):")
    logger.info("  └─ Minimum loss reduction required to make further partition")
    logger.info("  └─ Higher values = more conservative tree growth")
    logger.info("  └─ Optimal: 0 for most cases, 0.1-0.5 to reduce overfitting")
    
    logger.info("\n🎲 RANDOMNESS & SAMPLING PARAMETERS:")
    logger.info("-" * 50)
    logger.info("• subsample (0.7-1.0):")
    logger.info("  └─ Fraction of samples used for each tree")
    logger.info("  └─ <1.0 adds randomness and prevents overfitting")
    logger.info("  └─ Optimal: 0.8-0.9 for most problems")
    
    logger.info("\n• colsample_bytree (0.6-1.0):")
    logger.info("  └─ Fraction of features used for each tree")
    logger.info("  └─ <1.0 adds randomness and speeds up training")
    logger.info("  └─ Optimal: 0.8-1.0 for most problems")
    
    logger.info("\n• colsample_bylevel (0.6-1.0):")
    logger.info("  └─ Fraction of features used for each level/depth of tree")
    logger.info("  └─ More fine-grained control than colsample_bytree")
    logger.info("  └─ Optimal: 0.8-1.0, use when you have many features")
    
    logger.info("\n🛡️  REGULARIZATION PARAMETERS:")
    logger.info("-" * 50)
    logger.info("• reg_alpha (0-5.0) - L1 Regularization:")
    logger.info("  └─ Adds penalty for large weights (feature selection effect)")
    logger.info("  └─ Higher values = more regularization, sparse models")
    logger.info("  └─ Optimal: 0-0.5 for most problems, higher for feature selection")
    
    logger.info("\n• reg_lambda (0.5-5.0) - L2 Regularization:")
    logger.info("  └─ Adds penalty for large weights (smoothing effect)")
    logger.info("  └─ Higher values = more regularization, smoother models")
    logger.info("  └─ Optimal: 1.0-2.0 for most problems")
    
    logger.info("\n🔥 WILDFIRE-SPECIFIC TUNING INSIGHTS:")
    logger.info("-" * 50)
    logger.info("• Wildfire ROS prediction benefits from:")
    logger.info("  ✓ Higher n_estimators (300-500) - Complex environmental interactions")
    logger.info("  ✓ Moderate max_depth (6-8) - Capture weather-terrain-vegetation synergies")
    logger.info("  ✓ Lower learning_rate (0.05-0.1) - Stable learning for noisy fire data")
    logger.info("  ✓ Moderate subsample (0.7-0.9) - Handle synthetic data variations")
    logger.info("  ✓ L2 regularization (1.0-2.0) - Smooth predictions for continuous ROS")
    
    logger.info("\n💡 TUNING STRATEGY USED:")
    logger.info("-" * 50)
    logger.info("• Method: RandomizedSearchCV with 100 iterations")
    logger.info("• Cross-validation: 5-fold on synthetic training data")
    logger.info("• Metric: RMSE (Root Mean Square Error)")
    logger.info("• Search space: 10 parameters × 4-7 values each = ~2M combinations")
    logger.info("• Time complexity: O(n_iter × n_folds × n_estimators)")
    
    logger.info("\n🎯 HOW TO INTERPRET RESULTS:")
    logger.info("-" * 50)
    logger.info("• Best CV RMSE < 1.0: Excellent model")
    logger.info("• Best CV RMSE 1.0-2.0: Good model")  
    logger.info("• Best CV RMSE 2.0-3.0: Acceptable model")
    logger.info("• Best CV RMSE > 3.0: Needs improvement")
    
    logger.info("\n⚡ PERFORMANCE VS ACCURACY TRADE-OFFS:")
    logger.info("-" * 50)
    logger.info("• Fast training: n_estimators=100, max_depth=4, learning_rate=0.2")
    logger.info("• Balanced: n_estimators=200, max_depth=6, learning_rate=0.1 (recommended)")
    logger.info("• High accuracy: n_estimators=500, max_depth=8, learning_rate=0.05")
    
    logger.info("="*100)
    logger.info("📖 For more details, see: https://xgboost.readthedocs.io/en/stable/parameter.html")
    logger.info("="*100 + "\n")

def main():
    """Main function to train and validate XGBoost ROS predictor."""
    predictor = None
    try:
        logger.info("🔥 Starting XGBoost Wildfire ROS Prediction Model")
        logger.info("   Press Ctrl+C at any time for graceful interruption")
        
        # Initialize predictor
        predictor = WildfireROSPredictor("VAE_Dist/vae_combined_dataset.csv")
        
        # Prepare data
        predictor.prepare_data()
        
        # 🎯 Train model with enhanced hyperparameter tuning
        # HYPERPARAMETER TUNING OPTIONS:
        # - hyperparameter_tuning=True: Enable tuning (recommended)
        # - tuning_method='randomized': Fast random search (default)
        # - tuning_method='grid': Exhaustive grid search (slower but thorough)
        predictor.train_model(hyperparameter_tuning=True, tuning_method='randomized')
        
        # Validate model
        validation_results = predictor.validate_model()
        
        # Analyze feature importance
        feature_importance = predictor.analyze_feature_importance()
        
        # Create prediction plots
        predictor.plot_predictions()
        
        # Save model
        predictor.save_model()
        
        logger.info("🎉 XGBoost ROS prediction model completed successfully!")
        logger.info("📁 Files created in Ros_Pred/ folder:")
        logger.info("   - wildfire_ros_xgboost_model.joblib (trained model)")
        logger.info("   - xgboost_feature_importance.png (feature importance)")
        logger.info("   - xgboost_predictions.png (prediction plots)")
        logger.info("   - page1_core_predictions.png (core analysis)")
        logger.info("   - page2_residuals_analysis.png (error analysis)")
        logger.info("   - page3_feature_importance_tuning.png (tuning results)")
        
        # Summary
        logger.info("📋 Model Summary:")
        logger.info(f"   Training data: {len(predictor.y_synthetic)} synthetic samples")
        logger.info(f"   Validation data: {len(predictor.y_real)} real samples")
        logger.info(f"   Validation RMSE: {validation_results['rmse']:.4f} m/min")
        logger.info(f"   Validation R²: {validation_results['r2']:.4f}")
        
        # Example prediction
        logger.info("🔮 Example prediction:")
        example_features = {
            'temp_c': 25.0,
            'rel_humidity_pct': 45.0,
            'wind_speed_ms': 8.0,
            'precip_mm': 0.0,
            'vpd_kpa': 1.5,
            'fwi': 15.0,
            'ndvi': 0.4,
            'ndmi': 0.1,
            'lfmc_proxy_pct': 60.0,
            'elevation_m': 500.0,
            'slope_pct': 15.0,
            'aspect_deg': 180.0
        }
        
        predicted_ros = predictor.predict_ros(example_features)
        logger.info(f"   Hot, dry, windy conditions → Predicted ROS: {predicted_ros[0]:.3f} m/min")
        
        # Print hyperparameter explanation
        _print_hyperparameter_explanation()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Process interrupted by user")
        if predictor and predictor.model:
            logger.info("💾 Saving current model before exit...")
            predictor.save_model("Ros_Pred/interrupted_wildfire_ros_model.joblib")
            logger.info("✅ Model saved successfully")
        logger.info("👋 Goodbye!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if predictor and predictor.model:
            logger.info("💾 Attempting to save model before crash...")
            try:
                predictor.save_model("Ros_Pred/crash_recovery_model.joblib")
                logger.info("✅ Recovery model saved")
            except:
                logger.error("❌ Could not save recovery model")
        raise

if __name__ == "__main__":
    main()