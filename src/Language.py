import json
from pathlib import Path
from typing import Tuple
from unicodedata import normalize

import numpy as np
from sklearn.model_selection import train_test_split


class Language:
    SOS_ID = 0
    SOS_TOKEN = 'SOS'
    EOS_ID = 1
    EOS_TOKEN = 'EOS'
    PAD_ID = 2
    PAD_TOKEN = 'PAD'

    def __init__(self, name, sep=None):
        self.name = name
        self.token2index = {
            self.SOS_TOKEN: self.SOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.PAD_TOKEN: self.PAD_ID
        }
        self.token2count = {}
        # self.index2token = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.index2token = {
            self.SOS_ID: self.SOS_TOKEN,
            self.EOS_ID: self.EOS_TOKEN,
            self.PAD_ID: self.PAD_TOKEN
        }
        self.n_tokens = 3
        self.max_length = 0
        self.sep = sep

    @staticmethod
    def normalize(s):
        return normalize('NFKC', s)

    def sent_iter(self, sentence):
        if self.sep:
            return sentence.split(self.sep)
        else:
            return sentence

    def add_sentence(self, sentence):
        sentence = self.normalize(sentence)
        iterator_ = self.sent_iter(sentence)

        for token in iterator_:
            self.add_token(token)

        self.max_length = max(self.max_length, len(iterator_))

    def add_token(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

    @classmethod
    def read_data_from_txt(
            cls,
            data_path: str | Path,
            max_length=75,
    ) -> Tuple[np.array, np.array, "Language", "Language"]:
        if isinstance(data_path, str):
            data_path = Path(data_path)
        elif not isinstance(data_path, Path):
            raise ValueError("data_path must be a string or a Path object")

        assert data_path.exists(), f"Data path {data_path} does not exist"

        with data_path.open("r") as f:
            pairs = [cls.normalize(line.strip()).split("\t") for line in f if line.strip()]

        pairs = [
            (
                p0, p1
            )
            for p0, p1 in pairs
            if len(p0) <= max_length
            and len(p1) <= max_length
        ]

        l1 = cls('1')
        l2 = cls('2', sep=' | ')

        for pair in pairs:
            l1.add_sentence(pair[0])
            l2.add_sentence(pair[1])

        X, y = zip(
            *[
                (
                    [cls.SOS_ID] +
                    [
                        l1.token2index[token] for token in l1.sent_iter(pair[0])
                    ] + [cls.EOS_ID] + [cls.PAD_ID] * (l1.max_length - len(l1.sent_iter(pair[0]))),
                    [cls.SOS_ID] +
                    [
                        l2.token2index[token] for token in l2.sent_iter(pair[1])
                    ] + [cls.EOS_ID] + [cls.PAD_ID] * (l2.max_length - len(l2.sent_iter(pair[1])))
                )
                for pair in pairs
                # if (len_pair0 := len(pair[0])) <= max_length
                # and (len_pair1 := len(pair[1])) <= max_length
            ]
        )

        print(len(X), len(y))
        print(X[0], y[0])
        print(len(X[0]), len(y[0]))

        X = np.array(X)
        y = np.array(y)
        return X, y, l1, l2

    @classmethod
    def load_data(
            cls,
            X_path: str | Path,
            y_path: str | Path,
            lang_path: str | Path,
    ) -> Tuple[np.array, np.array, "Language", "Language"]:
        if isinstance(X_path, str):
            X_path = Path(X_path)
        elif not isinstance(X_path, Path):
            raise ValueError("X_path must be a string or a Path object")

        if isinstance(y_path, str):
            y_path = Path(y_path)
        elif not isinstance(y_path, Path):
            raise ValueError("y_path must be a string or a Path object")

        if isinstance(lang_path, str):
            lang_path = Path(lang_path)
        elif not isinstance(lang_path, Path):
            raise ValueError("lang_path must be a string or a Path object")

        assert X_path.exists(), f"X path {X_path} does not exist"
        assert y_path.exists(), f"y path {y_path} does not exist"
        assert lang_path.exists(), f"Language path {lang_path} does not exist"

        X, y = np.load(X_path), np.load(y_path)

        with open(lang_path, 'r') as f:
            lang = json.load(f)

        lang['1']['token2index'] = {k: int(v) for k, v in lang['1']['token2index'].items()}
        lang['1']['index2token'] = {int(k): v for k, v in lang['1']['index2token'].items()}
        lang['2']['token2index'] = {k: int(v) for k, v in lang['2']['token2index'].items()}
        lang['2']['index2token'] = {int(k): v for k, v in lang['2']['index2token'].items()}

        l1 = cls('1')
        l2 = cls('2')
        l1.__dict__.update(lang['1'])
        l2.__dict__.update(lang['2'])

        return X, y, l1, l2

    @classmethod
    def save_data(
            cls,
            X: np.array,
            y: np.array,
            l1: "Language",
            l2: "Language",
            X_path: str | Path,
            y_path: str | Path,
            lang_path: str | Path,
    ) -> None:
        if isinstance(X_path, str):
            X_path = Path(X_path)
        elif not isinstance(X_path, Path):
            raise ValueError("X_path must be a string or a Path object")

        if isinstance(y_path, str):
            y_path = Path(y_path)
        elif not isinstance(y_path, Path):
            raise ValueError("y_path must be a string or a Path object")

        if isinstance(lang_path, str):
            lang_path = Path(lang_path)
        elif not isinstance(lang_path, Path):
            raise ValueError("lang_path must be a string or a Path object")

        np.save(X_path, X)
        np.save(y_path, y)

        with open(lang_path, 'w') as f:
            json.dump({'1': l1.__dict__, '2': l2.__dict__}, f, ensure_ascii=False, indent=4)


def read_data(x_path: str | Path = 'X.npy', y_path: str | Path = 'y.npy', lang_path: str | Path = 'lang.json'):
    if isinstance(x_path, str):
        x_path = Path(x_path)
    elif not isinstance(x_path, Path):
        raise ValueError("x_path must be a string or a Path object")

    if isinstance(y_path, str):
        y_path = Path(y_path)
    elif not isinstance(y_path, Path):
        raise ValueError("y_path must be a string or a Path object")

    if isinstance(lang_path, str):
        lang_path = Path(lang_path)
    elif not isinstance(lang_path, Path):
        raise ValueError("lang_path must be a string or a Path object")

    assert x_path.exists(), f"X path {x_path} does not exist"
    assert y_path.exists(), f"y path {y_path} does not exist"
    assert lang_path.exists(), f"Language path {lang_path} does not exist"

    X, y, lang_input, lang_output = Language.load_data(x_path, y_path, lang_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test, lang_input, lang_output

if __name__ == '__main__':
    pho = True

    x_data = 'X_pho.npy' if pho else 'X.npy'
    y_data = 'y_pho.npy' if pho else 'y.npy'
    lang_path = 'lang_pho.json' if pho else 'lang.json'
    og_lang_path = 'all_noyeaux_pho.txt' if pho else 'all_noyeaux.txt'


    X, y, l1, l2 = Language.read_data_from_txt(og_lang_path)
    Language.save_data(X, y, l1, l2, x_data, y_data, lang_path)
