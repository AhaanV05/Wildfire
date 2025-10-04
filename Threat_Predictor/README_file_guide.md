# 📁 Project File Organization Guide

## 🗂️ Folder Structure

```
Wildfire/Threat_Predictor/
├── Ros_Pred/                          # XGBoost ROS Prediction Results
│   ├── wildfire_ros_xgboost_model.joblib    # Trained XGBoost model
│   ├── page1_core_predictions.png           # Core prediction analysis
│   ├── page2_residuals_analysis.png         # Error and residuals analysis
│   ├── page3_feature_importance_tuning.png  # Feature importance & tuning
│   ├── xgboost_feature_importance.png       # Feature importance chart
│   └── xgboost_predictions.png              # Actual vs predicted plots
│
├── VAE_Dist/                          # VAE Data Generation Results
│   ├── vae_combined_dataset.csv             # 3,000 samples (140 real + 2,860 synthetic)
│   ├── vae_data_comparison.png              # Real vs synthetic comparison
│   ├── best_vae_model.pth                   # Best trained VAE model
│   └── vae_checkpoint_epoch_*.pth           # Training checkpoints every 100 epochs
│
├── vae_generator.py                   # VAE synthetic data generator
├── xgboost_ros_predictor.py          # XGBoost ROS prediction model
├── Cleaned_ros_features.csv          # Original 140 cleaned samples
└── README_file_guide.md              # This file
```

## 🔍 What is a .joblib file?

### **Definition:**
A `.joblib` file is a **serialized Python object** saved using the `joblib` library, which is optimized for scientific computing and machine learning models.

### **Key Features:**
- **Efficient**: Faster than Python's built-in `pickle` for large NumPy arrays
- **Compressed**: Automatically compresses large models to save disk space
- **Cross-platform**: Works across different operating systems
- **ML-optimized**: Specifically designed for scikit-learn and ML workflows

### **What's Inside Our .joblib File:**
```python
model_data = {
    'model': self.model,                    # Trained XGBoost regressor
    'scaler': self.scaler,                  # StandardScaler for features
    'feature_columns': self.feature_columns, # List of feature names
    'target_column': self.target_column     # Target variable name
}
```

### **How to Load and Use:**
```python
import joblib

# Load the model
model_data = joblib.load('Ros_Pred/wildfire_ros_xgboost_model.joblib')
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']

# Make predictions
new_features = [[25.0, 45.0, 8.0, ...]]  # Your feature values
scaled_features = scaler.transform(new_features)
predicted_ros = model.predict(scaled_features)
```

### **Advantages over other formats:**
| Format | Size | Speed | ML Support | Cross-platform |
|--------|------|-------|------------|----------------|
| .joblib | ✅ Small | ✅ Fast | ✅ Excellent | ✅ Yes |
| .pickle | ❌ Large | ❌ Slow | ⚠️ Basic | ✅ Yes |
| .json | ❌ Very Large | ❌ Very Slow | ❌ None | ✅ Yes |
| .csv | ❌ Huge | ❌ Very Slow | ❌ None | ✅ Yes |

## 📊 File Contents Breakdown

### **Ros_Pred/ Folder:**
- **Model file (.joblib)**: Complete trained XGBoost model ready for production
- **Visualization pages**: Professional analysis charts split for clarity
- **Performance metrics**: Comprehensive model evaluation results

### **VAE_Dist/ Folder:**
- **Dataset (.csv)**: High-quality synthetic wildfire data for training
- **Model checkpoints (.pth)**: PyTorch VAE models at different training stages
- **Comparison plots**: Visual validation of synthetic vs real data quality

## 🚀 Usage Examples

### **1. Load Model for Predictions:**
```python
import joblib
import numpy as np

# Load model
model_data = joblib.load('Ros_Pred/wildfire_ros_xgboost_model.joblib')
model = model_data['model']
scaler = model_data['scaler']

# Predict wildfire ROS
fire_conditions = np.array([[
    25.0,  # temp_c
    45.0,  # rel_humidity_pct
    8.0,   # wind_speed_ms
    0.0,   # precip_mm
    1.5,   # vpd_kpa
    15.0,  # fwi
    0.4,   # ndvi
    0.1,   # ndmi
    60.0,  # lfmc_proxy_pct
    500.0, # elevation_m
    15.0,  # slope_pct
    180.0  # aspect_deg
]])

scaled_features = scaler.transform(fire_conditions)
ros_prediction = model.predict(scaled_features)
print(f"Predicted ROS: {ros_prediction[0]:.3f} m/min")
```

### **2. Load VAE for More Data Generation:**
```python
import torch
from vae_generator import ConditionalVAE

# Load VAE model
vae = ConditionalVAE(feature_dim=15, condition_dim=4, latent_dim=12, hidden_dim=128)
vae.load_state_dict(torch.load('VAE_Dist/best_vae_model.pth'))

# Generate more synthetic samples
# (requires additional setup - see vae_generator.py)
```

## 🔧 Technical Details

### **File Sizes (Approximate):**
- `.joblib` model: ~1-5 MB (compressed ML model)
- `.pth` VAE model: ~100-500 KB (neural network weights)
- `.csv` dataset: ~500 KB (3,000 samples × 16 features)
- `.png` visualizations: ~500 KB each (high-resolution plots)

### **Compatibility:**
- **Python versions**: 3.7+
- **Required libraries**: joblib, xgboost, scikit-learn, torch (for VAE)
- **Operating systems**: Windows, Linux, macOS

### **Model Performance:**
- **Training data**: 2,860 VAE-generated synthetic samples
- **Validation data**: 140 real wildfire samples  
- **RMSE**: ~2.6 m/min on real data validation
- **R²**: ~0.33 (explains 33% of variance)
- **Features**: 12 environmental variables (weather, vegetation, terrain)

---
*Generated by Wildfire ROS Prediction System - October 2025*