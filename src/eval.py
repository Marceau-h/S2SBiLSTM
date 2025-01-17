import re
from random import sample
from statistics import mean

from jiwer import process_words, visualize_alignment
from torch import device as torch_device
from torch.cuda import is_available
from tqdm.auto import trange, tqdm

try:
    import cowsay
except ImportError:
    cowsay = None

multi_stars = re.compile(r"\*+")

def dont(x):
    """
    Just don't (identity function)
    """
    return x


def align_words(ref, hyp):
    if ref == hyp:
        return None, 0

    computed = process_words([ref], [hyp], dont, dont)
    alignement = visualize_alignment(computed, show_measures=False)
    _, ref, hyp, *_ = multi_stars.sub("@", alignement).split("\n")
    alignement = f"{ref[5:]}\n{hyp[5:]}".replace(" ", " | ")

    return alignement, computed.wer

def predict(model, input_sentence, lang_input, lang_output, device=None):
    device = device or torch_device("cuda" if is_available() else "cpu")

    input_sentence_lst = [lang_input.index2token[token] for token in input_sentence if token != lang_input.PAD_ID]
    predicted_output_lst = model.predict(
        input_sentence, max_len=lang_output.n_tokens + 1, device=device, lang_output=lang_output
    )

    return input_sentence_lst, predicted_output_lst

def do_one_sent(model, sentence, lang_input, lang_output, device=None):
    device = device or torch_device("cuda" if is_available() else "cpu")

    input_sentence = [lang_input.SOS_ID] + [lang_input.token2index[token] for token in lang_input.sent_iter(sentence)] + [lang_input.EOS_ID]

    input_sentence_lst, predicted_output_lst = predict(model, input_sentence, lang_input, lang_output, device)

    line1 = f"Input sentence: {sentence}"
    line2 = f"Predicted output: {' | '.join(predicted_output_lst)}"
    max_len = max(len(line1), len(line2))
    txt = f"{line1:^{max_len}}\n{line2:^{max_len}}"

    if cowsay is not None:
        cowsay.tux(txt)
    else:
        print("\n\t\t------------------\t\t\n\n" + txt + "\n\n\t\t------------------\t\t\n")

    return input_sentence_lst, predicted_output_lst


def core_eval(X_test, y_test, lang_input, lang_output, model, nb_predictions=None, device=None):
    if nb_predictions is None:
        pbar = trange(len(X_test), desc="Evaluating", unit="sentence")
    elif isinstance(nb_predictions, int)  and nb_predictions > 0:
        pbar = tqdm(sample(range(len(X_test)), nb_predictions), desc="Evaluating", unit="sentence")
    else:
        raise ValueError("nb_predictions must be a positive integer or None")

    res = []
    for i in pbar:
        input_sentence = X_test[i]
        target_output = y_test[i]

        input_sentence_lst, predicted_output_lst = predict(model, input_sentence, lang_input, lang_output, device)
        target_output_lst = [lang_output.index2token[token] for token in target_output if token != lang_output.PAD_ID]

        exact_match = target_output_lst == predicted_output_lst

        aligned, wer = align_words(target_output_lst, predicted_output_lst)

        res.append((input_sentence_lst, target_output_lst, predicted_output_lst, aligned, exact_match, wer))

    return res


def random_predict(X_test, y_test, lang_input, lang_output, model, device=None, print_output=True, nb_predictions=10):
    device = device or torch_device("cuda" if is_available() else "cpu")

    res = core_eval(X_test, y_test, lang_input, lang_output, model, nb_predictions, device)

    if print_output:
        for input_sentence_lst, target_output_lst, predicted_output_lst, aligned, exact_match, wer in res:
            print(f"""
Lengths (\\wo EOS) - Input: {len(input_sentence_lst) - 6}, Target: {len(target_output_lst) - 2}, Predicted: {len(predicted_output_lst) - 2}

Input sentence: {"".join(input_sentence_lst)}
Target output: {" | ".join(target_output_lst)}
Predicted output: {" | ".join(predicted_output_lst)}

Alignment:\n{aligned}

Exact match: {exact_match}
WER: {wer:.2f}
""")
        if nb_predictions > 1:
            print(f"Mean exact match ratio: {mean(r[-2] for r in res):.3f}")
            print(f"Mean WER: {mean(r[-1] for r in res):.3f}")

    return res


def evaluate(X_test, y_test, lang_input, lang_output, model, device=None):
    device = device or torch_device("cuda" if is_available() else "cpu")

    res = core_eval(X_test, y_test, lang_input, lang_output, model, None, device)

    exact_match = mean(r[-2] for r in res)

    print(f"Exact match ratio: {exact_match:.3f}")

    wer_score = mean(r[-1] for r in res)

    print(f"Mean WER: {wer_score:.3f}")

    res_for_save = [
        {
            "input": "".join(r[0]),
            "target": " | ".join(r[1]),
            "predicted": " | ".join(r[2]),
            "alignment": r[3],
            "target_length": len(r[1]) - 2,
            "predicted_length": len(r[2]) - 2,
            "exact_match": r[4],
            "wer": r[5]
        }
        for r in res
    ]

    return res, exact_match, wer_score, res_for_save


def do_full_eval(X_test, y_test, lang_input, lang_output, model, eval_path, device):
    res, accuracy, wer_score, res_for_save = evaluate(X_test, y_test, lang_input, lang_output, model, device=device)

    import polars as pl

    df = pl.DataFrame(res_for_save)
    df.write_csv(eval_path.with_suffix(".csv"))
    df.write_ndjson(eval_path.with_suffix(".ndjson"))
    df.write_parquet(eval_path.with_suffix(".parquet"))
    df.write_json(eval_path.with_suffix(".json"))
