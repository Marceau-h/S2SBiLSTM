import torch
from torch import nn
import numpy as np
from huggingface_hub import PyTorchModelHubMixin

class S2SBiLSTM(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://huggingface.co/Marceau-H/S2SBiLSTM",
    language="French",
    license="AGPL-3.0",
):
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
