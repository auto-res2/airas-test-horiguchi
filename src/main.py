import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.preprocess import get_dataloader
from src.train import NetQ, NetG, NetD, LWGAN, IALWGAN, train_model
from src.evaluate import evaluate_lambda, compute_distance_correlation, latent_interpolation, plot_interpolations, plot_loss_curves, plot_hyperparameter_results


def experiment_1(device):
    print("\nStarting Experiment 1: Ablation Study on the Isometric Regularizer")
    dataloader = get_dataloader(batch_size=64, train=True)
    
    z_dim = 100
    # Instantiate networks for both models
    netQ_full = NetQ(z_dim)
    netG_full = NetG(z_dim)
    netD_full = NetD()
    full_model = IALWGAN(z_dim, netQ_full, netG_full, netD_full, device=device)

    netQ_base = NetQ(z_dim)
    netG_base = NetG(z_dim)
    netD_base = NetD()
    baseline_model = IALWGAN(z_dim, netQ_base, netG_base, netD_base, device=device)

    optimizer_full = torch.optim.Adam(full_model.parameters(), lr=0.0002)
    optimizer_baseline = torch.optim.Adam(baseline_model.parameters(), lr=0.0002)

    print("Training Full Model with isometric regularizer (λ_iso = 1.0)")
    loss_history_full = train_model(full_model, dataloader, optimizer_full, num_epochs=5, lambda_iso=1.0)

    print("Training Baseline Model without isometric regularizer (λ_iso = 0.0)")
    loss_history_baseline = train_model(baseline_model, dataloader, optimizer_baseline, num_epochs=5, lambda_iso=0.0)

    plot_loss_curves(loss_history_full, loss_history_baseline, filename="training_loss_ablation.pdf")


def experiment_2(device):
    print("\nStarting Experiment 2: Hyperparameter Sensitivity Analysis for Isometric Loss Weight (λ₂)")
    dataloader = get_dataloader(batch_size=64, train=True)
    
    lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    results = {}
    
    for lam in lambda_values:
        history, mse, corr = evaluate_lambda(lam, IALWGAN, dataloader, device, NetQ, NetG, NetD)
        results[lam] = {'loss_history': history, 'recon_error': mse, 'distance_corr': corr}
        print(f"λ_iso: {lam}, Reconstruction Error: {mse:.4f}, Distance Correlation: {corr:.4f}")

    lambdas = list(results.keys())
    recon_errors = [results[l]['recon_error'] for l in lambdas]
    correlations = [results[l]['distance_corr'] for l in lambdas]

    plot_hyperparameter_results(lambdas, recon_errors, correlations, filename="loss_vs_lambda.pdf")


def experiment_3(device):
    print("\nStarting Experiment 3: Visual and Quantitative Evaluation of Latent Space Geometry")
    dataloader = get_dataloader(batch_size=64, train=True)

    z_dim = 100
    # Full Model (with isometric regularizer)
    netQ_full = NetQ(z_dim)
    netG_full = NetG(z_dim)
    netD_full = NetD()
    full_model = IALWGAN(z_dim, netQ_full, netG_full, netD_full, device=device)
    optimizer_full = torch.optim.Adam(full_model.parameters(), lr=0.0002)
    train_model(full_model, dataloader, optimizer_full, num_epochs=3, lambda_iso=1.0)

    # Baseline Model (without isometric regularizer)
    netQ_base = NetQ(z_dim)
    netG_base = NetG(z_dim)
    netD_base = NetD()
    baseline_model = IALWGAN(z_dim, netQ_base, netG_base, netD_base, device=device)
    optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=0.0002)
    train_model(baseline_model, dataloader, optimizer_base, num_epochs=3, lambda_iso=0.0)

    # Extract embeddings for evaluation (using 200 samples)
    def extract_embeddings(model, dataloader, num_samples=200):
        model.eval()
        embeddings = []
        images = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(model.device)
                emb = model.netQ(data, rank=0)
                embeddings.append(emb.cpu())
                images.append(data.cpu())
                if sum(x.size(0) for x in embeddings) >= num_samples:
                    break
        embeddings = torch.cat(embeddings, dim=0)[:num_samples]
        images = torch.cat(images, dim=0)[:num_samples]
        return embeddings, images

    embeddings_full, _ = extract_embeddings(full_model, dataloader, num_samples=200)
    embeddings_base, _ = extract_embeddings(baseline_model, dataloader, num_samples=200)

    corr_full = compute_distance_correlation(embeddings_full)
    corr_base = compute_distance_correlation(embeddings_base)
    print(f"Latent-Distance Correlation (Full Model): {corr_full:.4f}")
    print(f"Latent-Distance Correlation (Baseline): {corr_base:.4f}")

    # Latent space interpolation
    sample_imgs, _ = next(iter(dataloader))
    img1, img2 = sample_imgs[0], sample_imgs[1]

    interpolations_full = latent_interpolation(full_model, img1, img2, steps=10, device=device)
    interpolations_base = latent_interpolation(baseline_model, img1, img2, steps=10, device=device)

    plot_interpolations(interpolations_full, title="Full Model Latent Interpolations", filename="interpolation_full.pdf")
    plot_interpolations(interpolations_base, title="Baseline Model Latent Interpolations", filename="interpolation_baseline.pdf")


def test_code():
    print("\n===== Starting Test Run =====")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    experiment_1(device)
    experiment_2(device)
    experiment_3(device)
    
    print("Test run finished. Check the .research/iteration1/images directory for generated PDF plots.")


if __name__ == '__main__':
    test_code()
