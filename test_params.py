from random import sample, seed

from src.Language import read_data
from src.eval import evaluate
from src.train import auto_train
from src.model import S2SBiLSTM, paths

seed(42)

def test_sample(
        X_test,
        y_test,
        eval_size=50,
):
    which_ones = sample(range(len(X_test)), eval_size)

    X_sample = X_test[which_ones]
    y_sample = y_test[which_ones]
    
    return X_sample, y_sample
        


def test_epochs(
        num_epochs,
        X_test,
        y_test,
        eval_size=50,
        step=10,
        **train_args
):
    
    X_sample, y_sample = test_sample(X_test, y_test, eval_size)

    _, _, _, _, losses, evals, _ = auto_train(
        **train_args,
        num_epochs=num_epochs,
        eval_every=step, 
        eval_fn=evaluate, 
        eval_args={
            "X_test": X_sample,
            "y_test": y_sample,
            "do_print": False,
        },
    )
    return losses, [e[2] for e in evals]


def test_embed_size(
        embed_sizes,
        eval_size=50,
        **train_args
):
    losses = []
    wers = []

    for embed_size in embed_sizes:
        model, lang_input, lang_output, params, losses, evals, (X_train, X_test, y_train, y_test) = auto_train(
            **train_args,
            embed_size=embed_size
        )

        X_sample, y_sample = test_sample(X_test, y_test, eval_size)

        res, exact_match, wer_score, res_for_save = evaluate(X_sample, y_sample, lang_input, lang_output, model)

        wers.append(wer_score)
        losses.append(losses[-1])

    return losses, wers


def test_hidden_size(
        hidden_sizes,
        eval_size=50,
        **train_args
):
    losses = []
    wers = []

    for hidden_size in hidden_sizes:
        model, lang_input, lang_output, params, losses, evals, (X_train, X_test, y_train, y_test) = auto_train(
            **train_args,
            hidden_size=hidden_size
        )

        X_sample, y_sample = test_sample(X_test, y_test, eval_size)

        res, exact_match, wer_score, res_for_save = evaluate(X_sample, y_sample, lang_input, lang_output, model)

        wers.append(wer_score)
        losses.append(losses[-1])

    return losses, wers

def test_num_layers(
        num_layers,
        eval_size=50,
        **train_args
):
    losses = []
    wers = []

    for num_layer in num_layers:
        model, lang_input, lang_output, params, losses, evals, (X_train, X_test, y_train, y_test) = auto_train(
            **train_args,
            num_layers=num_layer
        )

        X_sample, y_sample = test_sample(X_test, y_test, eval_size)

        res, exact_match, wer_score, res_for_save = evaluate(X_sample, y_sample, lang_input, lang_output, model)

        wers.append(wer_score)
        losses.append(losses[-1])

    return losses, wers


def test_lr(
        lrs,
        eval_size=50,
        **train_args
):
    losses = []
    wers = []

    for lr in lrs:
        model, lang_input, lang_output, params, losses, evals, (X_train, X_test, y_train, y_test) = auto_train(
            **train_args,
            lr=lr
        )

        X_sample, y_sample = test_sample(X_test, y_test, eval_size)

        res, exact_match, wer_score, res_for_save = evaluate(X_sample, y_sample, lang_input, lang_output, model)

        wers.append(wer_score)
        losses.append(losses[-1])

    return losses, wers

def main(
        num=10,
        mode="epochs",
        pho=True,
        eval_size=100,
):
    params_path, model_path, og_lang_path, x_data, y_data, lang_path, eval_path = paths(pho)

    range_ = [2**i for i in range(num)]

    train_args = {
        "x_data": x_data,
        "y_data": y_data,
        "lang_path": lang_path,
    }

    match mode:
        case "epochs":
            X_train, X_test, y_train, y_test, lang_input, lang_output = read_data(
                x_data,
                y_data,
                lang_path,
            )

            losses, evals = test_epochs(
                num,
                X_test,
                y_test,
                eval_size=eval_size,
                step=1,
                **train_args,
            )
        case "embed_size":
            losses, evals = test_embed_size(
                range_,
                eval_size=eval_size,
                **train_args,
            )
        case "hidden_size":
            losses, evals = test_hidden_size(
                range_,
                eval_size=eval_size,
                **train_args,
            )

        case "num_layers":
            losses, evals = test_num_layers(
                range_,
                eval_size=eval_size,
                **train_args,
            )

        case "lr":
            range_ = [1/(10**i) for i in range(num)]
            losses, evals = test_lr(
                range_,
                eval_size=eval_size,
                **train_args,
            )
        case _:
            raise ValueError("Invalid mode")

    print(losses)
    print(evals)

    import plotly.express as px

    range_ = range_ if mode != "epochs" else range(1, num+1)
    for data, data_name in ((losses, "Loss"), (evals, "WER")):
        if mode != "epochs" and data_name == "Loss":
            continue

        title = f"{mode} vs {data_name}"
        fig = px.line(x=range_, y=data, labels={"x": mode, "y": data_name}, title=title)

        if mode == "lr":
            fig.update_xaxes(type="log")
            fig.update_layout(xaxis_autorange="reversed")

        fig.write_html(f"{title}.html")
        fig.write_image(f"{title}.png")
        fig.write_image(f"{title}.svg")
        # fig.write_image(f"{title}.eps")

        try:
            fig.show()
        except:
            pass

    return losses, evals


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--mode", type=str, default="epochs")
    parser.add_argument("--pho", action="store_true")
    parser.add_argument("--eval_size", type=int, default=100)
    args = parser.parse_args()

    main(**vars(args))

