import torch


def train_autoencoder(
        model,
        train_loader,
        optimizer,
        loss_fn,
        epochs=10,
        missingness=None,
        corruption_type='mcar',
        image_indices=None,
        device="cpu",
        **corruption_kwargs
):
    model.to(device)
    model.train()
    losses = []
    images = {}

    # Optional: track reconstruction during training
    if image_indices is not None:
        for idx in image_indices:
            img = train_loader.dataset[idx][0]  # shape: (1, 28, 28)
            images[idx] = {"original": img.unsqueeze(0).to(device), "reconstructions": []}

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        for x, _ in train_loader:
            x = x.to(device)

            if missingness is not None:
                noisy_x, mask = missingness.apply_corruption(x, corruption_type, **corruption_kwargs)
                # noisy_x = torch.nan_to_num(noisy_x, nan=0.0)
                noisy_x = noisy_x.to(device)
            else:
                noisy_x = x

            x_hat, z = model(noisy_x)
            loss = loss_fn(x_hat, x.view(x.size(0), -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)

        # Optional: add reconstruction after each epoch
        if image_indices is not None:
            model.eval()
            with torch.no_grad():
                for k, v in images.items():
                    original = v["original"]
                    original_flat = original.view(original.size(0), -1)
                    x_hat, _ = model(original_flat)
                    images[k]["reconstructions"].append(x_hat.cpu())

            model.train()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}: average loss = {epoch_loss:.4f}")

    return (losses, images) if image_indices is not None else losses
