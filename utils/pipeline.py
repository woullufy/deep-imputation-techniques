import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import optim
from torch.nn import KLDivLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader

from models import Autoencoder, DEC
from .training_ae import train_autoencoder
from .training_dec import train_dec


def run_dec_pipeline(
        X_clean,
        y_true,
        data_indices,
        missingness=None,
        imputer=None,
        device='cpu',
        ae_epochs=20,
        dec_epochs=50,
        n_clusters=10,
        latent_dim=10,
        n_features=784,
        **corruption_kwargs,
):
    # ----- Corrupting clean data -----
    missing_rate = corruption_kwargs.get("missing_rate", 0)
    corruption_type = corruption_kwargs.get("corruption_type", "mcar")

    print(f"Corruption settings ({corruption_type} | {missing_rate:.2f})")

    if imputer is not None and missing_rate > 0:
        X_corrupted_flat, _ = missingness.apply_corruption(X_clean, **corruption_kwargs)
    else:
        X_corrupted_flat = X_clean.clone()

    # ----- Impute into corrupted data -----
    if imputer is not None:
        print(f'\tRunning imputation: {imputer.__class__.__name__}')

        # Reshape image for imputer
        H = W = int(n_features ** 0.5)
        X_img = X_corrupted_flat.view(-1, 1, H, W)

        # Apply the imputation
        X_imputed_img = imputer.impute(X_img)
        X_final_flat = X_imputed_img.view(-1, n_features)
    else:
        X_final_flat = X_corrupted_flat

    # ----- Autoencoder training -----
    print('\t- Training Autoencoder')

    tensor_x = X_final_flat.to(device)
    dataset = TensorDataset(tensor_x, data_indices)

    ae_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    dec_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    ae = Autoencoder(input_dim=n_features, latent_dim=latent_dim).to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=0.001)
    ae_loss_fn = MSELoss()

    train_autoencoder(
        model=ae,
        train_loader=ae_loader,
        optimizer=ae_optimizer,
        loss_fn=ae_loss_fn,
        epochs=ae_epochs,
        missingness=missingness if imputer is None else None,
        device=device,
        **corruption_kwargs,
    )

    # ----- DEC training -----
    print('\t- Training DEC')
    dec = DEC(autoencoder=ae, num_clusters=n_clusters, latent_dim=latent_dim).to(device)
    dec_optimizer = optim.SGD(dec.parameters(), lr=0.01, momentum=0.9)
    dec_loss_fn = KLDivLoss(reduction='batchmean')

    train_dec(
        model=dec,
        train_loader=dec_loader,
        optimizer=dec_optimizer,
        loss_fn=dec_loss_fn,
        tensor_x=tensor_x,
        epochs=dec_epochs,
        device=device
    )

    # ----- Evaluation -----
    print('\tEvaluation')
    dec.eval()
    with torch.no_grad():
        q, _ = dec(tensor_x)
        y_pred = torch.argmax(q, dim=1).cpu().numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    print(f'Result: ARI={ari:.4f} | NMI={nmi:.4f}')
    return ari, nmi
