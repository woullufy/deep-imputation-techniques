import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from torch import nn
from torch import optim
from torch.nn import MSELoss, KLDivLoss
from torch.utils.data import TensorDataset, DataLoader

from models import Autoencoder
from models import TabularAutoencoder, DEC
from models.gmm import GMMMissing
from utils import clustering_accuracy
from .training_ae import train_autoencoder
from .training_ae import train_tabular_autoencoder
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
    corruption_type = corruption_kwargs.get("corruption_type")
    value = corruption_kwargs.get("missing_rate", 0)
    value = corruption_kwargs.get("num_rows", 0)

    print(f"Corruption settings ({corruption_type} | {value:.2f})")

    if (imputer is not None) and (missingness is not None):
        X_corrupted_flat, _ = missingness.apply_corruption(X_clean, **corruption_kwargs)
    else:
        print("\tNo corruption applied")
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
    acc = clustering_accuracy(y_true, y_pred)

    print(f'Result: ARI={ari:.4f} | NMI={nmi:.4f} | ACC={acc:.4f}')
    return ari, nmi, acc


def run_tabular_dec_pipeline(
        X_clean,
        y_true,
        data_indices,
        missingness=None,
        imputer=None,
        device='cpu',
        ae_epochs=200,
        dec_epochs=100,
        n_clusters=3,
        latent_dim=2,
        **corruption_kwargs
):
    # ----- 1. Handle Corruption & Imputation -----
    method = corruption_kwargs.get("method", "mcar")
    frac = corruption_kwargs.get("frac", 0.0)
    print(f"Starting Tabular DEC Pipeline | Corruption: {method} ({frac:.2f})")

    if missingness is not None:
        # Move to CPU for missingness logic
        X_corrupted_np = missingness.apply_stratified(X_clean.cpu(), y_true, **corruption_kwargs)
        X_input_tensor = torch.from_numpy(X_corrupted_np).float().to(device)
    else:
        X_input_tensor = X_clean.to(device)

    # External Imputer vs DAE logic
    if imputer is not None:
        print(f"\t- Applying Imputation: {imputer.__class__.__name__}")
        X_final_np = imputer.fit_transform(X_input_tensor.cpu().numpy())
        X_input_tensor = torch.from_numpy(X_final_np).float().to(device)
        current_missingness = None
    else:
        print("\t- Denoising Autoencoder mode")
        current_missingness = missingness

    # ----- 2. Autoencoder training -----
    dataset = TensorDataset(X_input_tensor, data_indices)
    batch_size = 16 if len(y_true) < 1000 else 128
    ae_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_features = X_clean.shape[1]
    ae = TabularAutoencoder(input_dim=n_features, latent_dim=latent_dim).to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=0.001)
    ae_loss_fn = nn.MSELoss()

    train_tabular_autoencoder(
        model=ae, train_loader=ae_loader, optimizer=ae_optimizer,
        loss_fn=ae_loss_fn, epochs=ae_epochs, device=device,
        missingness=current_missingness, **corruption_kwargs
    )

    # ----- 3. HEALING STEP (Crucial Fix for KMeans NaN Error) -----
    print('\t- Healing data with trained AE for DEC phase')
    ae.eval()
    with torch.no_grad():
        # Passing input through AE returns x_hat (reconstructed/clean data)
        X_healed, _ = ae(X_input_tensor)

        # Update loaders to use the HEALED data (No more NaNs)
    dec_dataset = TensorDataset(X_healed, data_indices)
    dec_loader = DataLoader(dec_dataset, batch_size=batch_size, shuffle=False)

    # ----- 4. DEC training -----
    print('\t- Training DEC')
    dec = DEC(autoencoder=ae, num_clusters=n_clusters, latent_dim=latent_dim).to(device)
    dec_optimizer = optim.SGD(dec.parameters(), lr=0.01, momentum=0.9)
    dec_loss_fn = nn.KLDivLoss(reduction='batchmean')

    train_dec(
        model=dec,
        train_loader=dec_loader,  # Now receives healed data: KMeans won't crash
        optimizer=dec_optimizer,
        loss_fn=dec_loss_fn,
        tensor_x=X_healed,  # Now receives healed data: KLDiv won't crash
        epochs=dec_epochs,
        device=device
    )

    # ----- 5. Evaluation -----
    print('\t- Evaluation')
    dec.eval()
    with torch.no_grad():
        q, _ = dec(X_healed)
        y_pred = torch.argmax(q, dim=1).cpu().numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    acc = clustering_accuracy(y_true, y_pred)

    print(f'Result: ARI={ari:.4f} | NMI={nmi:.4f} | ACC={acc:.4f}')
    return ari, nmi, acc

# def run_tabular_dec_pipeline(
#         X_clean,
#         y_true,
#         data_indices,
#         missingness=None,
#         imputer=None,  # Sklearn imputer (optional)
#         device='cpu',
#         ae_epochs=50,
#         dec_epochs=100,
#         n_clusters=3,
#         latent_dim=2,
#         **corruption_kwargs
# ):
#     print(f"Starting Tabular DEC Pipeline | Latent: {latent_dim} | Clusters: {n_clusters}")
#
#     # ----- 1. Handle Corruption & Imputation -----
#     # Convert clean tensor to numpy for the TabularMissingness logic
#     X_clean_np = X_clean.detach().cpu().numpy()
#
#     if missingness is not None:
#         # Inject missingness (results in NaNs)
#         X_corrupted_np = missingness.apply_stratified(X_clean_np, y_true, **corruption_kwargs)
#
#         if imputer is not None:
#             # External Imputation (e.g., Mean, MICE, kNN)
#             print(f"\t- Applying Imputation: {imputer.__class__.__name__}")
#             X_final_np = imputer.fit_transform(X_corrupted_np)
#             X_input_tensor = torch.from_numpy(X_final_np).float().to(device)
#             current_missingness = None  # Already filled, AE acts as regular AE
#         else:
#             # No imputer: DAE logic (AE will fill NaNs with 0.0 internally)
#             print("\t- No imputer provided: Using Denoising Autoencoder logic")
#             X_input_tensor = torch.from_numpy(X_corrupted_np).float().to(device)
#             current_missingness = missingness
#     else:
#         print("\t- No corruption applied")
#         X_input_tensor = X_clean.to(device)
#         current_missingness = None
#
#     # ----- 2. Setup DataLoaders -----
#     dataset = TensorDataset(X_input_tensor, data_indices)
#     batch_size = 32 if len(y_true) < 1000 else 128
#
#     ae_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     dec_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#
#     # ----- 3. Train Tabular Autoencoder -----
#     print('\t- Training Tabular Autoencoder')
#     ae = TabularAutoencoder(input_dim=X_clean.shape[1], latent_dim=latent_dim).to(device)
#     ae_optimizer = optim.Adam(ae.parameters(), lr=0.001)
#     ae_loss_fn = MSELoss()
#
#     train_tabular_autoencoder(
#         model=ae,
#         train_loader=ae_loader,
#         optimizer=ae_optimizer,
#         loss_fn=ae_loss_fn,
#         epochs=ae_epochs,
#         missingness=current_missingness if imputer is None else None,
#         device=device,
#         **corruption_kwargs
#     )
#
#     # ----- 4. Train DEC -----
#     print('\t- Training DEC')
#     dec = DEC(autoencoder=ae, num_clusters=n_clusters, latent_dim=latent_dim).to(device)
#     dec_optimizer = optim.SGD(dec.parameters(), lr=0.01, momentum=0.9)
#     dec_loss_fn = KLDivLoss(reduction='batchmean')
#
#     train_dec(
#         model=dec,
#         train_loader=dec_loader,
#         optimizer=dec_optimizer,
#         loss_fn=dec_loss_fn,
#         tensor_x=X_input_tensor,
#         epochs=dec_epochs,
#         device=device
#     )
#
#     # ----- 5. Evaluation -----
#     print('\t- Evaluation')
#     dec.eval()
#     with torch.no_grad():
#         q, _ = dec(X_input_tensor)
#         y_pred = torch.argmax(q, dim=1).cpu().numpy()
#
#     ari = adjusted_rand_score(y_true, y_pred)
#     nmi = normalized_mutual_info_score(y_true, y_pred)
#     acc = clustering_accuracy(y_true, y_pred)
#
#     print(f'Result: ARI={ari:.4f} | NMI={nmi:.4f} | ACC={acc:.4f}')
#     return ari, nmi, acc


def run_gmm_pipeline(
        frac_list,
        X,
        y,
        n_classes,
        missingness,
        imputer=None,
        missing_method="mar"
):
    run_ari, run_nmi, run_acc = [], [], []

    for frac in frac_list:
        if missing_method == "mcar":
            missing_data = missingness.apply_stratified(
                X, y, method="mcar", frac=frac
            )
        else:
            missing_data = missingness.apply_stratified(
                X, y, method="mar", frac=frac, dep_col=0, miss_col=1
            )

        if imputer is not None:
            x_filled = imputer.fit_transform(missing_data)
            gmm = GaussianMixture(n_components=n_classes)
            y_pred = gmm.fit_predict(x_filled)
        else:
            gmm = GMMMissing(n_components=n_classes)
            gmm.fit(missing_data)
            y_pred = gmm.predict(missing_data)

        run_ari.append(adjusted_rand_score(y, y_pred))
        run_nmi.append(normalized_mutual_info_score(y, y_pred))
        run_acc.append(clustering_accuracy(y, y_pred))

    return run_ari, run_nmi, run_acc
