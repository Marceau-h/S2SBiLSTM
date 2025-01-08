from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.Language import read_data
from src.model import S2SBiLSTM


def train(model, dataloader, optimizer, criterion, device, num_epochs=10, teacher_forcing_ratio=0.5):
    model.train()

    pbar = trange(1, num_epochs + 1, desc="Epochs", unit="epoch")
    for epoch in pbar:
        epoch_loss = 0

        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg, teacher_forcing_ratio)

            # Reshape for the loss function
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        pbar.set_postfix(loss=epoch_loss / len(dataloader))


def auto_train(
        num_epochs: int = 10,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        lr: float = 1e-3,
        batch_size: int = 512,
        teacher_forcing_ratio: float = 0.5,
        save_root: str | Path = None,
        x_data: str | Path = 'X.npy',
        y_data: str | Path = 'y.npy',
        lang_path: str | Path = 'lang.json',
        device=None
):
    if isinstance(save_root, str):
        save_root = Path(save_root)
    elif not isinstance(save_root, Path) and save_root is not None:
        raise ValueError("save_root must be a string or a Path object")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read data
    X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(
        x_data,
        y_data,
        lang_path,
    )

    # Model setup
    input_size = lang_input.n_tokens
    output_size = lang_output.n_tokens

    # Initialize model
    model = S2SBiLSTM(input_size, output_size, embed_size, hidden_size, num_layers).to(device)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming padding index is 0

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    train(
        model,
        dataloader,
        optimizer,
        criterion,
        device=device,
        num_epochs=num_epochs,
        teacher_forcing_ratio=teacher_forcing_ratio
    )

    params = {
        "num_epochs": num_epochs,
        "input_size": input_size,
        "output_size": output_size,
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": lr
        },
        "criterion": "CrossEntropyLoss",
        "criterion_parameters": {
            "ignore_index": 0
        },
        "batch_size": batch_size,
        "teacher_forcing_ratio": teacher_forcing_ratio,
    }

    return model, lang_input, lang_output, params, (X_train, X_test, y_train, y_test)
