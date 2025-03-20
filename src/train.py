from pathlib import Path
from typing import Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.Language import read_data
from src.model import S2SBiLSTM


def train(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        num_epochs=10,
        teacher_forcing_ratio=0.5,
        eval_every=None,
        eval_fn=None,
        eval_args=None,
        from_epoch=0,
):
    model.train()

    losses = []
    evals = []

    pbar = trange(1 + from_epoch, num_epochs + 1 + from_epoch, desc="Epochs", unit="epoch")
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

        if eval_every and eval_fn:
            if epoch % eval_every == 0:
                losses.append(epoch_loss / len(dataloader))
                evals.append(eval_fn(**eval_args))
                model.train()

    if not eval_every:
        losses.append(epoch_loss / len(dataloader))
    elif epoch % eval_every != 0:
        losses.append(epoch_loss / len(dataloader))
        if eval_fn:
            evals.append(eval_fn(**eval_args))

    return model, losses, evals


def expand_model_vocabulary(model, new_src_vocab_size, new_trg_vocab_size, device=None):
    """Expand model embedding layers to accommodate larger vocabularies."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get original embedding dimensions
    old_src_vocab_size = model.encoder_embedding.num_embeddings
    old_trg_vocab_size = model.decoder_embedding.num_embeddings
    embed_dim = model.encoder_embedding.embedding_dim

    # Create new embeddings with expanded size
    new_encoder_embed = nn.Embedding(new_src_vocab_size, embed_dim)
    new_decoder_embed = nn.Embedding(new_trg_vocab_size, embed_dim)

    # Initialize with normal distribution or zeros
    nn.init.normal_(new_encoder_embed.weight, mean=0, std=0.1)
    nn.init.normal_(new_decoder_embed.weight, mean=0, std=0.1)

    # Copy original embeddings to new ones
    with torch.no_grad():
        new_encoder_embed.weight[:old_src_vocab_size] = model.encoder_embedding.weight
        new_decoder_embed.weight[:old_trg_vocab_size] = model.decoder_embedding.weight

    # Replace embeddings in the model
    model.encoder_embedding = new_encoder_embed
    model.decoder_embedding = new_decoder_embed

    # If there's an output projection layer that depends on vocab size
    if hasattr(model, 'fc_out') and isinstance(model.fc_out, nn.Linear):
        old_fc = model.fc_out
        new_fc = nn.Linear(old_fc.in_features, new_trg_vocab_size)

        # Copy original weights for existing vocab
        with torch.no_grad():
            new_fc.weight[:old_trg_vocab_size] = old_fc.weight
            new_fc.bias[:old_trg_vocab_size] = old_fc.bias

        model.fc_out = new_fc

    return model.to(device)

def auto_train(
        num_epochs: int = 10,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        lr: float = 1e-3,
        batch_size: int = 512,
        teacher_forcing_ratio: float = 0.5,
        save_root: str | Path = None,
        x_data: str | Path = 'X.npy',
        y_data: str | Path = 'y.npy',
        lang_path: str | Path = 'lang.json',
        eval_every: Optional[int] = None,
        eval_fn: "function" = None,
        eval_args: dict = None,
        device=None,
        from_=None,
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

    model = S2SBiLSTM(input_size, output_size, embed_size, hidden_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming padding index is 0

    if from_ is not None:
        model, state, old_vocab_size = from_
        # model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        criterion.load_state_dict(state["criterion_state_dict"])
        # num_epochs += state["epoch"]

        if input_size != old_vocab_size:
            model = expand_model_vocabulary(model, input_size, output_size, device=device)
    else:
        state = None

    # Training setup

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train model
    model, losses, evals = train(
        model,
        dataloader,
        optimizer,
        criterion,
        device=device,
        num_epochs=num_epochs,
        teacher_forcing_ratio=teacher_forcing_ratio,
        eval_every=eval_every,
        eval_fn=eval_fn,
        eval_args={
            **(eval_args or {}),
            "lang_input": lang_input,
            "lang_output": lang_output,
            "model": model,
        }
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

    state = {
        "epoch": num_epochs + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "loss": losses[-1],
    }

    return model, lang_input, lang_output, (params, state), losses, evals, (X_train, X_test, y_train, y_test)
