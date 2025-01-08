import json
from pathlib import Path

import torch

from Language import Language, read_data
from eval import random_predict, evaluate
from model import S2SBiLSTM
from train import auto_train

if __name__ == '__main__':
    do_train = True
    pho = True
    make_lang = False
    full_eval = False

    num_epochs = 10
    embed_size = 512
    hidden_size = 512
    num_layers = 1
    lr = 1e-4
    batch_size = 512
    teacher_forcing_ratio = 0.5
    nb_predictions = 10

    params_path = "params_pho.json" if pho else "params.json"
    model_path = "model_pho.pth" if pho else "model.pth"
    og_lang_path = 'all_noyeaux_pho.txt' if pho else 'all_noyeaux.txt'
    x_data = 'X_pho.npy' if pho else 'X.npy'
    y_data = 'y_pho.npy' if pho else 'y.npy'
    lang_path = 'lang_pho.json' if pho else 'lang.json'
    eval_path = Path('results_pho.json' if pho else 'results.json')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if make_lang:
        X, y, l1, l2 = Language.read_data_from_txt(og_lang_path)
        Language.save_data(X, y, l1, l2, x_data, y_data, lang_path)

    if do_train:
        (
            model,
            lang_input,
            lang_output,
            params,
            (X_train, X_test, y_train, y_test)
        ) = auto_train(
            num_epochs=num_epochs,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            lr=lr,
            batch_size=batch_size,
            teacher_forcing_ratio=teacher_forcing_ratio,
            x_data=x_data,
            y_data=y_data,
            lang_path=lang_path,
            device=device
        )

        print(params)

        torch.save(model.state_dict(), model_path)

        params["model_path"] = model_path

        with open(params_path, "w") as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

        print("Model and parameters saved successfully")

    else:
        # Load model and parameters
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
                weights_only=True
            )
        )

        X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(x_data, y_data, lang_path)
        print("Model, data, and parameters loaded successfully")

    # Test prediction
    random_predict(X_test, y_test, lang_input, lang_output, model, device=device, nb_predictions=nb_predictions)

    if full_eval:
        res, accuracy, wer_score, res_for_save = evaluate(X_test, y_test, lang_input, lang_output, model, device=device)

        import polars as pl

        df = pl.DataFrame(res_for_save)
        df.write_csv(eval_path.with_suffix(".csv"))
        df.write_ndjson(eval_path.with_suffix(".ndjson"))
        df.write_parquet(eval_path.with_suffix(".parquet"))
        df.write_json(eval_path.with_suffix(".json"))
