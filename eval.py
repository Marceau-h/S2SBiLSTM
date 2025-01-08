import re
from random import choice

from torch import device as torch_device
from torch.cuda import is_available
from jiwer import process_words, visualize_alignment
from tqdm.auto import trange

multi_stars = re.compile(r"\*{2,}")

def align_words(ref, hyp):
    if ref == hyp:
        return None, 0

    computed = process_words([ref], [hyp], lambda x : x, lambda x : x)
    alignement = visualize_alignment(computed, show_measures=False)
    _, ref, hyp, *_ = multi_stars.sub("@", alignement).split("\n")
    alignement = f"{ref[5:]}\n{hyp[5:]}".replace(" ", " | ")

    return alignement, computed.wer


def random_predict(X_test, y_test, lang_input, lang_output, model, device=None, print_output=True, nb_predictions=10):
    device = device or torch_device("cuda" if is_available() else "cpu")

    res = []
    for _ in range(nb_predictions):
        random_idx = choice(range(len(X_test)))
        random_sentence = X_test[random_idx]
        random_target = y_test[random_idx]

        input_sentence_lst = (
            [lang_input.index2token[token] for token in random_sentence if token != lang_input.PAD_ID])
        target_output_lst = ([lang_output.index2token[token] for token in random_target if token != lang_output.PAD_ID])

        predicted_output_lst = model.predict(random_sentence, max_len=lang_output.n_tokens + 1, device=device, lang_output=lang_output)

        accuracy_score = sum([1 for i, j in zip(target_output_lst, predicted_output_lst) if i == j]) / len(
            target_output_lst)

        aligned, wer = align_words(target_output_lst, predicted_output_lst)


        if print_output:
            print(f"""
Lengths (\\wo EOS) - Input: {len(input_sentence_lst) - 3}, Target: {len(target_output_lst) - 1}, Predicted: {len(predicted_output_lst) - 1}

Input sentence: {"".join(input_sentence_lst)}
Target output: {" | ".join(target_output_lst)}
Predicted output: {" | ".join(predicted_output_lst)}

Alignment:\n{aligned}

Accuracy: {accuracy_score:.2f}
WER: {wer:.2f}
""")

        res.append((input_sentence_lst, target_output_lst, predicted_output_lst, aligned, accuracy_score, wer))

    return res


def evaluate(X_test, y_test, lang_input, lang_output, model, device=None):
    device = device or torch_device("cuda" if is_available() else "cpu")

    res = []
    res_for_save = []
    pbar = trange(len(X_test), desc="Evaluating", unit="sentence")
    for i in pbar:
        input_sentence = X_test[i]
        target_output = y_test[i]

        input_sentence_lst = (
            [lang_input.index2token[token] for token in input_sentence if token != lang_input.PAD_ID])
        target_output_lst = ([lang_output.index2token[token] for token in target_output if token != lang_output.PAD_ID])

        predicted_output_lst = model.predict(input_sentence, max_len=lang_output.n_tokens + 1, device=device, lang_output=lang_output)

        accuracy_score = sum([1 for i, j in zip(target_output_lst, predicted_output_lst) if i == j]) / len(
            target_output_lst)

        aligned, wer = align_words(target_output_lst, predicted_output_lst)

        res.append((input_sentence_lst, target_output_lst, predicted_output_lst, aligned, accuracy_score, wer))
        res_for_save.append({
            "input": "".join(input_sentence_lst),
            "target": " | ".join(target_output_lst),
            "predicted": " | ".join(predicted_output_lst),
            "alignment": aligned,
            "target_length": len(target_output_lst) ,
            "predicted_length": len(predicted_output_lst),
            "accuracy": accuracy_score,
            "wer": wer
        }
        )

    accuracy_scores = [r[-2] for r in res]
    accuracy = sum(accuracy_scores) / len(accuracy_scores)

    print(f"Accuracy: {accuracy:.2f}")

    wer_scores = [r[-1] for r in res]
    wer_score = sum(wer_scores) / len(wer_scores)

    print(f"WER: {wer_score:.2f}")

    return res, accuracy, wer_score, res_for_save
