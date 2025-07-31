import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
from scipy.stats import pearsonr


def evaluate_lambda(lambda_iso_value, model_constructor, dataloader, device, NetQ, NetG, NetD):
    """ 
    Train one instance of IALWGAN with a given lambda_iso value and return metrics.
    """
    z_dim = 100
    netQ = NetQ(z_dim)
    netG = NetG(z_dim)
    netD = NetD()
    model = model_constructor(z_dim, netQ, netG, netD, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    print(f"Training with λ_iso = {lambda_iso_value}")
    # Train for a few epochs (demo with 3 epochs)
    from src.train import train_model
    loss_history = train_model(model, dataloader, optimizer, num_epochs=3, lambda_iso=lambda_iso_value)

    # Compute reconstruction error on one batch
    model.eval()
    mse_loss = nn.MSELoss(reduction='sum')
    recon_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            rec = model.netG(model.netQ(data, rank=0))
            recon_error += mse_loss(data, rec).item()
            total_samples += data.size(0)
            break  # only one batch for quick demo
    recon_error /= total_samples

    # Compute latent-to-data pairwise distance correlation on one batch
    batch_data, _ = next(iter(dataloader))
    batch_data = batch_data.to(device)
    with torch.no_grad():
        latent_vectors = model.netQ(batch_data, rank=0)
    batch_flat = batch_data.view(batch_data.size(0), -1)
    latent_dists = torch.cdist(latent_vectors, latent_vectors, p=2).cpu().numpy().flatten()
    data_dists = torch.cdist(batch_flat, batch_flat, p=2).cpu().numpy().flatten()
    r, _ = pearsonr(latent_dists, data_dists)

    return loss_history, recon_error, r


def compute_distance_correlation(embeddings, feature_output=None):
    """
    Compute the Pearson correlation between pairwise distances computed from the embeddings and feature_output.
    If feature_output is None, embeddings are used.
    """
    if feature_output is None:
        feature_output = embeddings
    latent_dists = torch.cdist(embeddings, embeddings, p=2).numpy().flatten()
    feature_dists = torch.cdist(feature_output, feature_output, p=2).numpy().flatten()
    r, _ = pearsonr(latent_dists, feature_dists)
    return r


def latent_interpolation(model, img1, img2, steps=10, device=torch.device('cpu')):
    model.eval()
    interpolated = []
    with torch.no_grad():
        z1 = model.netQ(img1.unsqueeze(0).to(device), rank=0)
        z2 = model.netQ(img2.unsqueeze(0).to(device), rank=0)
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            generated = model.netG(z_interp)
            interpolated.append(generated.squeeze(0).cpu())
    return interpolated


def plot_interpolations(interpolations, title="Interpolation", filename="interpolation.pdf"):
    from torchvision import utils
    import os
    # Create directory if not exists
    import os
    out_dir = os.path.join('.research', 'iteration1', 'images')
    os.makedirs(out_dir, exist_ok=True)
    
    grid = utils.make_grid(torch.stack(interpolations), nrow=len(interpolations), normalize=True, scale_each=True)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved interpolation plot as {filepath}")
    plt.close()


def plot_loss_curves(loss_history_full, loss_history_baseline, filename="training_loss_ablation.pdf"):
    import os
    out_dir = os.path.join('.research', 'iteration1', 'images')
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure()
    plt.plot(loss_history_full, label="Full Model")
    plt.plot(loss_history_baseline, label="Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Loss Curve Comparison for Ablation Study")
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved loss curve plot as {filepath}")
    plt.close()


def plot_hyperparameter_results(lambdas, recon_errors, correlations, filename="loss_vs_lambda.pdf"):
    import os
    out_dir = os.path.join('.research', 'iteration1', 'images')
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lambdas, recon_errors, marker='o')
    plt.xlabel("λ₂ (isometric loss weight)")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("Reconstruction Error vs. λ₂")
    
    plt.subplot(1, 2, 2)
    plt.plot(lambdas, correlations, marker='x', color='red')
    plt.xlabel("λ₂ (isometric loss weight)")
    plt.ylabel("Latent-Data Distance Correlation")
    plt.title("Distance Correlation vs. λ₂")
    plt.tight_layout()
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved hyperparameter sensitivity plots as {filepath}")
    plt.close()
