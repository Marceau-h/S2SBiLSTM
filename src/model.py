import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn, optim


class S2SBiLSTM(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://huggingface.co/Marceau-H/S2SBiLSTM",
    language="French",
    license="AGPL-3.0",
):
    @staticmethod
    def jsonify_types(obj):
        if isinstance(obj, Path):
            return obj.as_posix()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def __init__(self, input_size, output_size, embed_size, hidden_size, num_layers=1):
        super(S2SBiLSTM, self).__init__()

        # Encoder components
        self.encoder_embedding = nn.Embedding(input_size, embed_size)
        self.encoder_lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        # Decoder components
        self.decoder_embedding = nn.Embedding(output_size, embed_size)
        self.decoder_lstm = nn.LSTM(
            embed_size, hidden_size * 2,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        trg_vocab_size = self.fc.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)

        # Encode the source sequence
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

        # Concatenate the forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)

        # First input to the decoder is the <sos> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            embedded_trg = self.decoder_embedding(input).unsqueeze(1)

            # Decoder step
            output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            outputs[:, t, :] = prediction

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else prediction.argmax(1)

        return outputs

    def predict(self, src, max_len, device, lang_output):
        self.eval()
        if isinstance(src, (np.ndarray, list)):
            src = torch.tensor(src, device=device)
        else:
            src = src.to(device)

        # Encode the source sequence
        with torch.no_grad():
            embedded_src = self.encoder_embedding(src.unsqueeze(0))  # Add batch dimension
            encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)

            if len(hidden.shape) != 3:
                raise ValueError("Hidden shape is not 3D")

            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
            cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)

            # Initialize the decoder input with the <sos> token
            input = torch.tensor([lang_output.SOS_ID], device=device)

            outputs = [lang_output.SOS_ID]
            for _ in range(max_len):
                embedded_trg = self.decoder_embedding(input).unsqueeze(1)
                output, (hidden, cell) = self.decoder_lstm(embedded_trg, (hidden, cell))
                prediction = self.fc(output.squeeze(1))
                predicted_token = prediction.argmax(1).item()

                outputs.append(predicted_token)

                if predicted_token == lang_output.EOS_ID:
                    break

                input = torch.tensor([predicted_token], device=device)

        return [lang_output.index2token[token] for token in outputs]


def save_model(model, params, state, model_path, params_path):
    torch.save(model.state_dict(), model_path)

    params["model_path"] = model_path

    with open(params_path, "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=4, default=model.jsonify_types)

    torch.save(state, params_path.with_suffix(".state"))

    print("Model and parameters saved successfully")


def load_model(params_path, model_path, device):
    with open(params_path, "r") as f:
        params = json.load(f)

    print(params)

    model = S2SBiLSTM(
        params["input_size"],
        params["output_size"],
        params["embed_size"],
        params["hidden_size"],
        params["num_layers"]
    ).to(device)

    model.load_state_dict(
        torch.load(
            f=params.get("model_path", model_path),
            weights_only=False,
        )
    )

    state = torch.load(params_path.with_suffix(".state"), weights_only=False)
    # model.load_state_dict(state["model_state_dict"], strict=False,
    #
    # optimizer = optim.Adam(model.parameters(), lr=params["optimizer_parameters"]["lr"])
    # optimizer.load_state_dict(state["optimizer_state_dict"])
    #
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion.load_state_dict(state["criterion_state_dict"])

    old_vocab_size = model.encoder_embedding.weight.shape[1]

    return model, state, old_vocab_size


def paths(pho: bool = False, suffix: str = "", json_: bool = False) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    assert isinstance(pho, bool), "pho must be a boolean"
    assert isinstance(suffix, str), "suffix must be a string"

    if pho and not suffix:  # if pho is True and suffix is empty
        suffix = "_pho"

    relative_to_root = 0
    cwd = Path.cwd()
    while cwd.name != "S2SBiLSTM":
        relative_to_root += 1
        cwd = cwd.parent

    prepend = Path("../" * relative_to_root)

    params_path = prepend / f"params{suffix}.json"
    model_path = prepend / f"model{suffix}.pth"
    og_lang_path = prepend /  f"all_noyeaux{suffix}.txt"
    x_data = prepend / f"X{suffix}.npy"
    y_data = prepend / f"y{suffix}.npy"
    lang_path = prepend / f"lang{suffix}.json"
    eval_path = prepend / f"results{suffix}.json"
    json_data = prepend / f"caveau_flat{suffix}.json"

    return params_path, model_path, (og_lang_path if not json_ else json_data), x_data, y_data, lang_path, eval_path
