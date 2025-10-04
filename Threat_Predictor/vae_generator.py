#!/usr/bin/env python3
"""
Conditional Variational Autoencoder for Wildfire Dataset Generation

Advanced ML model that learns complex non-linear relationships between wildfire features
and generates realistic synthetic data conditioned on geographic and temporal factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🔥 Using device: {device}")

class WildfireDataset(Dataset):
    """Dataset class for wildfire features."""
    
    def __init__(self, features, conditions):
        self.features = torch.FloatTensor(features)
        self.conditions = torch.FloatTensor(conditions)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]

class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder for wildfire data generation."""
    
    def __init__(self, feature_dim=15, condition_dim=4, latent_dim=12, hidden_dim=128):
        super(ConditionalVAE, self).__init__()
        
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Deeper encoder with more capacity for extensive training
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Latent space parameterization
        self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)
        
        # Deeper decoder with more capacity for extensive training
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def encode(self, x, c):
        """Encode features and conditions to latent parameters."""
        combined = torch.cat([x, c], dim=1)
        h = self.encoder(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """Decode latent vector and conditions to features."""
        combined = torch.cat([z, c], dim=1)
        return self.decoder(combined)
    
    def forward(self, x, c):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar
    
    def generate(self, conditions, n_samples=1):
        """Generate new samples given conditions."""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim).to(device)
            conditions = conditions.to(device)
            generated = self.decode(z, conditions)
        return generated

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with KL divergence and reconstruction loss."""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

class WildfireVAEGenerator:
    """Main class for wildfire synthetic data generation using VAE."""
    
    def __init__(self, data_file="Cleaned_ros_features.csv"):
        """Initialize the VAE generator."""
        self.data_file = data_file
        self.real_data = pd.read_csv(data_file)
        
        # Define feature columns
        self.feature_columns = [
            'lat', 'lon', 'temp_c', 'rel_humidity_pct', 'wind_speed_ms', 'precip_mm',
            'vpd_kpa', 'fwi', 'ndvi', 'ndmi', 'lfmc_proxy_pct',
            'elevation_m', 'slope_pct', 'aspect_deg', 'target_ros_m_min'
        ]
        
        # Scalers for normalization
        self.feature_scaler = StandardScaler()
        self.condition_scaler = StandardScaler()
        
        logger.info(f"🔗 Loaded {len(self.real_data)} real samples")
        
    def prepare_data(self):
        """Prepare data for VAE training."""
        logger.info("🔧 Preparing data for VAE training...")
        
        # Clean data
        clean_data = self.real_data.dropna()
        logger.info(f"   Using {len(clean_data)} complete samples")
        
        # Extract features
        features = clean_data[self.feature_columns].values
        
        # Create conditions (geographic + temporal + elevation)
        clean_data['date'] = pd.to_datetime(clean_data['date'])
        clean_data['month'] = clean_data['date'].dt.month
        clean_data['day_of_year'] = clean_data['date'].dt.dayofyear
        
        # Normalize day_of_year to [0, 1]
        clean_data['season'] = np.sin(2 * np.pi * clean_data['day_of_year'] / 365.25)
        
        # Conditions: lat, lon, elevation, season
        conditions = clean_data[['lat', 'lon', 'elevation_m', 'season']].values
        
        # Normalize features and conditions
        features_normalized = self.feature_scaler.fit_transform(features)
        conditions_normalized = self.condition_scaler.fit_transform(conditions)
        
        self.clean_data = clean_data
        self.features_normalized = features_normalized
        self.conditions_normalized = conditions_normalized
        
        logger.info("✅ Data preparation completed")
        
    def train_vae(self, epochs=5000, batch_size=8, learning_rate=5e-4, target_loss=None, save_interval=100):
        """Train the VAE model with extensive epochs and data cycling."""
        logger.info("🧠 Training Conditional VAE with extensive data cycling...")
        logger.info(f"   Training for {epochs} epochs with batch size {batch_size}")
        logger.info(f"   Will cycle through {len(self.features_normalized)} samples multiple times per epoch")
        if target_loss:
            logger.info(f"   Will stop early if loss reaches {target_loss}")
        logger.info(f"   Model will be saved every {save_interval} epochs")
        logger.info("   🛑 Press Ctrl+C to stop training and save current best model")
        
        # Create dataset and dataloader with smaller batches for more iterations
        dataset = WildfireDataset(self.features_normalized, self.conditions_normalized)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        # Initialize deeper model for extensive training
        self.vae = ConditionalVAE(
            feature_dim=len(self.feature_columns),
            condition_dim=4,  # lat, lon, elevation, season
            latent_dim=12,  # Larger latent space for more complex patterns
            hidden_dim=128  # More capacity for extensive training
        ).to(device)
        
        # Optimizer with weight decay for better generalization
        optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.9)
        
        # Training loop with much more patience
        self.vae.train()
        train_losses = []
        best_loss = float('inf')
        patience = 500  # Much more patience for extensive training
        patience_counter = 0
        
        # Data cycling - repeat dataset multiple times per epoch for thorough learning
        cycles_per_epoch = 3  # Each epoch will see the data 3 times
        
        try:
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                
                # Multiple cycles through the data per epoch for thorough learning
                for cycle in range(cycles_per_epoch):
                    # Reshuffle data each cycle
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
                    
                    for batch_features, batch_conditions in dataloader:
                        batch_features = batch_features.to(device)
                        batch_conditions = batch_conditions.to(device)
                        
                        optimizer.zero_grad()
                        
                        # Forward pass
                        recon_features, mu, logvar = self.vae(batch_features, batch_conditions)
                        
                        # Progressive beta scheduling - slower ramp up for extensive training
                        beta = min(1.0, 0.05 + 0.95 * epoch / (epochs * 0.6))
                        loss = vae_loss(recon_features, batch_features, mu, logvar, beta=beta)
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                
                avg_loss = epoch_loss / batch_count
                train_losses.append(avg_loss)
                scheduler.step(avg_loss)
            
                # Early stopping and model saving
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.vae.state_dict(), 'best_vae_model.pth')
                    logger.info(f"   💾 New best model saved! Loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Save periodic checkpoints
                if epoch % save_interval == 0 and epoch > 0:
                    checkpoint_name = f'vae_checkpoint_epoch_{epoch}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.vae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'best_loss': best_loss
                    }, checkpoint_name)
                    logger.info(f"   💾 Checkpoint saved: {checkpoint_name}")
                
                # More frequent logging to monitor extensive training
                if epoch % 50 == 0:
                    logger.info(f"   Epoch {epoch:4d}: Loss = {avg_loss:.4f}, Beta = {beta:.3f}, Best = {best_loss:.4f}")
                    logger.info(f"          Batches/epoch: {batch_count}, Data seen: {batch_count * batch_size} samples")
                    logger.info(f"          Patience: {patience_counter}/{patience}")
                
                # Target loss stopping
                if target_loss and avg_loss <= target_loss:
                    logger.info(f"   🎯 Target loss {target_loss} reached! Stopping training.")
                    break
                
                if patience_counter >= patience:
                    logger.info(f"   ⏱️ Early stopping at epoch {epoch} (after {epoch * cycles_per_epoch * len(dataloader)} total iterations)")
                    break
                    
        except KeyboardInterrupt:
            logger.info(f"   🛑 Training interrupted by user at epoch {epoch}")
            logger.info(f"   💾 Current best model saved with loss: {best_loss:.4f}")
            # Make sure we have the best model loaded
            if Path('best_vae_model.pth').exists():
                self.vae.load_state_dict(torch.load('best_vae_model.pth'))
        
        # Load best model
        if Path('best_vae_model.pth').exists():
            self.vae.load_state_dict(torch.load('best_vae_model.pth'))
            logger.info(f"✅ Best model loaded with final loss: {best_loss:.4f}")
        
        logger.info("✅ VAE training completed")
        logger.info(f"   📊 Total epochs trained: {len(train_losses)}")
        logger.info(f"   📉 Final loss: {train_losses[-1]:.4f}")
        logger.info(f"   🏆 Best loss achieved: {best_loss:.4f}")
        return train_losses
        
    def generate_synthetic_samples(self, n_samples=2860):
        """Generate synthetic samples using trained VAE."""
        logger.info(f"🎲 Generating {n_samples} synthetic samples with VAE...")
        
        # Generate diverse conditions
        synthetic_conditions = self._generate_diverse_conditions(n_samples)
        
        # Normalize conditions
        conditions_normalized = self.condition_scaler.transform(synthetic_conditions)
        conditions_tensor = torch.FloatTensor(conditions_normalized)
        
        # Generate in batches to avoid memory issues
        batch_size = 100
        all_generated = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_conditions = conditions_tensor[i:end_idx]
            
            # Generate samples
            generated_batch = self.vae.generate(batch_conditions, len(batch_conditions))
            all_generated.append(generated_batch.cpu().numpy())
        
        # Combine all batches
        generated_features = np.vstack(all_generated)
        
        # Denormalize features
        generated_features = self.feature_scaler.inverse_transform(generated_features)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(generated_features, columns=self.feature_columns)
        
        # Add condition columns
        synthetic_df['lat'] = synthetic_conditions[:, 0]
        synthetic_df['lon'] = synthetic_conditions[:, 1]
        synthetic_df['elevation_m'] = synthetic_conditions[:, 2]
        
        # Apply physical constraints
        synthetic_df = self._apply_wildfire_constraints(synthetic_df)
        
        # Add metadata
        synthetic_df = self._add_vae_metadata(synthetic_df, synthetic_conditions)
        
        logger.info(f"✅ Generated {len(synthetic_df)} synthetic samples")
        return synthetic_df
    
    def _generate_diverse_conditions(self, n_samples):
        """Generate diverse geographic and temporal conditions."""
        # Use real data distribution as guide but with more diversity
        real_conditions = self.clean_data[['lat', 'lon', 'elevation_m', 'season']].values
        
        # Add some noise to create variations
        noise_scale = np.std(real_conditions, axis=0) * 0.3
        
        # Sample from real conditions with added noise
        indices = np.random.choice(len(real_conditions), n_samples, replace=True)
        synthetic_conditions = real_conditions[indices].copy()
        
        # Add controlled noise
        for i in range(synthetic_conditions.shape[1]):
            noise = np.random.normal(0, noise_scale[i], n_samples)
            synthetic_conditions[:, i] += noise
        
        # Ensure realistic bounds
        synthetic_conditions[:, 0] = np.clip(synthetic_conditions[:, 0], -90, 90)  # lat
        synthetic_conditions[:, 1] = np.clip(synthetic_conditions[:, 1], -180, 180)  # lon
        synthetic_conditions[:, 2] = np.clip(synthetic_conditions[:, 2], 0, 5000)  # elevation
        synthetic_conditions[:, 3] = np.clip(synthetic_conditions[:, 3], -1, 1)  # season
        
        return synthetic_conditions
    
    def _apply_wildfire_constraints(self, df):
        """Apply wildfire-specific physical constraints."""
        logger.info("🔧 Applying wildfire domain constraints...")
        
        # Physical bounds
        df['temp_c'] = np.clip(df['temp_c'], -40, 60)
        df['rel_humidity_pct'] = np.clip(df['rel_humidity_pct'], 0, 100)
        df['wind_speed_ms'] = np.clip(df['wind_speed_ms'], 0, 50)
        df['precip_mm'] = np.clip(df['precip_mm'], 0, 200)
        df['vpd_kpa'] = np.clip(df['vpd_kpa'], 0, 10)
        df['fwi'] = np.clip(df['fwi'], 0, 100)
        df['ndvi'] = np.clip(df['ndvi'], -1, 1)
        df['ndmi'] = np.clip(df['ndmi'], -1, 1)
        df['lfmc_proxy_pct'] = np.clip(df['lfmc_proxy_pct'], 0, 200)
        df['slope_pct'] = np.clip(df['slope_pct'], 0, 100)
        df['aspect_deg'] = np.clip(df['aspect_deg'], -1, 360)
        df['target_ros_m_min'] = np.clip(df['target_ros_m_min'], 0, 10)
        
        # Ensure VPD consistency with temperature and humidity
        for idx in df.index:
            temp = df.loc[idx, 'temp_c']
            rh = df.loc[idx, 'rel_humidity_pct']
            
            # Calculate VPD from temp and humidity
            es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
            ea = es * (rh / 100.0)
            vpd_calculated = max(es - ea, 0.0)
            df.loc[idx, 'vpd_kpa'] = vpd_calculated
        
        return df
    
    def _add_vae_metadata(self, df, conditions):
        """Add metadata to VAE-generated samples."""
        # Generate diverse dates based on season
        dates = []
        for season_val in conditions[:, 3]:  # season column
            # Convert season back to day of year
            day_of_year = int((np.arcsin(np.clip(season_val, -1, 1)) / (2 * np.pi) * 365.25) % 365.25)
            if day_of_year < 0:
                day_of_year += 365
            
            # Create date
            base_date = date(2024, 1, 1)
            sample_date = base_date + timedelta(days=day_of_year)
            
            # Adjust to fire season (May-October)
            if sample_date.month < 5:
                sample_date = sample_date.replace(month=np.random.randint(5, 11))
            elif sample_date.month > 10:
                sample_date = sample_date.replace(month=np.random.randint(5, 11))
            
            dates.append(sample_date)
        
        df['date'] = dates
        df['fire_id'] = [f"VAE_{i+1:06d}" for i in range(len(df))]
        
        return df
    
    def save_combined_dataset(self, synthetic_df, output_file="vae_combined_dataset.csv"):
        """Combine real and VAE-generated synthetic data."""
        logger.info("📝 Combining real and VAE synthetic data...")
        
        # Prepare real data
        real_prepared = self.real_data.copy()
        real_prepared['fire_id'] = [f"REAL_{i+1:06d}" for i in range(len(real_prepared))]
        
        # Ensure consistent columns
        all_columns = list(set(real_prepared.columns) | set(synthetic_df.columns))
        
        for col in all_columns:
            if col not in real_prepared.columns:
                real_prepared[col] = None
            if col not in synthetic_df.columns:
                synthetic_df[col] = None
        
        # Combine datasets
        combined_df = pd.concat([real_prepared, synthetic_df], ignore_index=True)
        
        # Save to CSV
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"💾 Saved VAE combined dataset to {output_file}")
        logger.info(f"   Real samples: {len(real_prepared)}")
        logger.info(f"   VAE synthetic samples: {len(synthetic_df)}")
        logger.info(f"   Total samples: {len(combined_df)}")
        
        return combined_df
    
    def plot_vae_comparison(self, synthetic_df, save_plots=True):
        """Plot comparison between real and VAE-generated synthetic data."""
        logger.info("📊 Creating VAE comparison plots...")
        
        # Select key features for plotting
        plot_features = ['temp_c', 'rel_humidity_pct', 'ndvi', 'elevation_m', 'target_ros_m_min']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(plot_features):
            if i < len(axes):
                ax = axes[i]
                
                # Plot distributions
                if feature in self.real_data.columns:
                    real_values = self.real_data[feature].dropna()
                    ax.hist(real_values, bins=30, alpha=0.7, 
                           label=f'Real (n={len(real_values)})', density=True, color='blue')
                
                if feature in synthetic_df.columns:
                    synthetic_values = synthetic_df[feature].dropna()
                    ax.hist(synthetic_values, bins=30, alpha=0.7, 
                           label=f'VAE Synthetic (n={len(synthetic_values)})', density=True, color='red')
                
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.legend()
                ax.set_title(f'{feature} Distribution Comparison')
                ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(plot_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('vae_data_comparison.png', dpi=300, bbox_inches='tight')
            logger.info("📊 Saved VAE comparison plot to vae_data_comparison.png")
        
        plt.show()

def main():
    """Main function to generate synthetic wildfire data using VAE."""
    try:
        # Initialize VAE generator
        logger.info("🔥 Starting VAE-based wildfire data generation...")
        generator = WildfireVAEGenerator("Cleaned_ros_features.csv")
        
        # Prepare data
        generator.prepare_data()
        
        # Train VAE with extensive epochs and data cycling
        # You can set target_loss to stop when satisfied (e.g., target_loss=50.0)
        train_losses = generator.train_vae(
            epochs=5000, 
            batch_size=8, 
            learning_rate=5e-4, 
            target_loss=None,  # Set to desired loss value to stop early
            save_interval=100   # Save checkpoints every 100 epochs
        )
        
        # Generate synthetic samples
        synthetic_data = generator.generate_synthetic_samples(n_samples=2860)
        
        # Save combined dataset
        combined_data = generator.save_combined_dataset(synthetic_data)
        
        # Create comparison plots
        generator.plot_vae_comparison(synthetic_data)
        
        logger.info("🎉 VAE synthetic data generation completed successfully!")
        logger.info("📁 Files created:")
        logger.info("   - vae_combined_dataset.csv (3,000+ total samples)")
        logger.info("   - vae_data_comparison.png (distribution comparison)")
        logger.info("   - best_vae_model.pth (trained VAE model)")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()