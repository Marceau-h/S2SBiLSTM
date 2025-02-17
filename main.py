from argparse import ArgumentParser

import torch

from src.Language import Language, read_data
from src.eval import random_predict, do_one_sent, do_full_eval
from src.model import save_model, load_model, paths
from src.train import auto_train
from sympy.strategies.branch import yieldify


def main(
        do_train: bool = False,
        pho: bool = True,
        suffix: str = "",
        make_lang: bool = False,
        json_lang: bool = False,
        full_eval: bool = False,

        num_epochs: int = 10,
        embed_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 1,
        lr: float = 1e-4,
        batch_size: int = 2048,
        teacher_forcing_ratio: float = 0.5,
        nb_predictions: int = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    params_path, model_path, og_lang_path, x_data, y_data, lang_path, eval_path = paths(pho, suffix)

    if make_lang:
        if json_lang:
            X, y, l1, l2 = Language.read_data_from_json(og_lang_path)
        else:
            X, y, l1, l2 = Language.read_data_from_txt(og_lang_path)
        Language.save_data(X, y, l1, l2, x_data, y_data, lang_path)

    if do_train:
        (
            model,
            lang_input,
            lang_output,
            params,
            losses,
            evals,
            (X_train, X_test, y_train, y_test),
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
            device=device,
        )

        print(params)

        save_model(model, params, model_path, params_path)

    else:
        model = load_model(params_path, model_path, device)

        X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(x_data, y_data, lang_path)
        print("Model, data, and parameters loaded successfully")

    # # push to hub
    # model.push_to_hub("Marceau-H/S2SBiLSTM")

    # # Save model architecture for schematic visualization
    # from torch.utils.data import DataLoader
    #
    # dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #
    # src, trg = next(iter(dataloader))
    # src, trg = src.to(device), trg.to(device)
    # output = model(src, trg, teacher_forcing_ratio)
    #
    # from torchviz import make_dot
    #
    # graph = make_dot(output, params=dict(list(model.named_parameters())))  # , show_attrs=True, show_saved=True)
    #
    # graph.render("model", format="png")
    #
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('runs/model_visualization')
    # writer.add_graph(model, [src, trg, teacher_forcing_ratio])
    # writer.close()

    # Test prediction
    random_predict(X_test, y_test, lang_input, lang_output, model, device=device, nb_predictions=nb_predictions)

    if full_eval:
        do_full_eval(X_test, y_test, lang_input, lang_output, model, eval_path, device=device)


def load_and_do_one_sent(sentence, pho):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params_path, model_path, og_lang_path, x_data, y_data, lang_path, eval_path = paths(pho)
    model = load_model(params_path, model_path, device)
    X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(x_data, y_data, lang_path)
    do_one_sent(model, sentence, lang_input, lang_output, device)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("sentence", type=str, help="Sentence to predict (will bypass all other arguments)", nargs="?", default=None)

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--pho", action="store_true", help="Use phonetic data")
    parser.add_argument("--make_lang", action="store_true", help="Make language data")
    parser.add_argument("--full_eval", action="store_true", help="Run full evaluation")

    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--embed_size", type=int, default=512, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="Teacher forcing ratio")
    parser.add_argument("--nb_predictions", type=int, default=10, help="Number of predictions to make")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for file names (overrides `--pho`)")

    args = parser.parse_args()

    if args.sentence:
        load_and_do_one_sent(args.sentence, pho=args.pho)
        exit(0)

    main(
        do_train=args.train,
        pho=args.pho,
        suffix=args.suffix,
        make_lang=args.make_lang,
        full_eval=args.full_eval,
        num_epochs=args.num_epochs,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        nb_predictions=args.nb_predictions
    )

    print("Done :)")
