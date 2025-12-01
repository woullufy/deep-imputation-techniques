import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn import KLDivLoss, MSELoss
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from models import *
from utils import *


def run_dec_pipeline(
        X_imputed_flat,
        y_true,
        data_indices,
        device='cpu',
        ae_epochs=50,
        dec_epochs=50,
        n_clusters=10,
        latent_dim=10,
        n_features=784,
):
    imputed_tensor_x = X_imputed_flat.to(device)
    imputed_dataset = TensorDataset(imputed_tensor_x, data_indices)
    imputed_loader = DataLoader(imputed_dataset, batch_size=256, shuffle=True)
    dec_loader = DataLoader(imputed_dataset, batch_size=256, shuffle=False)


    print("Autoencoder training")
    ae = Autoencoder(input_dim=n_features, latent_dim=latent_dim).to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=0.001)
    ae_loss_fn = MSELoss()

    train_autoencoder(
        model=ae,
        train_loader=imputed_loader,
        optimizer=ae_optimizer,
        loss_fn=ae_loss_fn,
        epochs=ae_epochs,
        missingness=None,
        device=device
    )

    print("DEC training")
    dec = DEC(autoencoder=ae, num_clusters=n_clusters, latent_dim=latent_dim).to(device)
    dec_optimizer = optim.SGD(dec.parameters(), lr=0.01, momentum=0.9)
    dec_loss_fn = KLDivLoss(reduction='batchmean')

    train_dec(
        model=dec,
        train_loader=imputed_loader,
        optimizer=dec_optimizer,
        loss_fn=dec_loss_fn,
        tensor_x=imputed_tensor_x,
        epochs=dec_epochs,
        device=device
    )

    # Prediction and evaluation
    dec.eval()
    with torch.no_grad():
        q, _ = dec(imputed_tensor_x)
        y_pred = torch.argmax(q, dim=1).cpu().numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return ari, nmi
