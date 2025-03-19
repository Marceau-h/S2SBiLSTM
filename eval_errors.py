from pathlib import Path

import polars as pl

def keep_errors_only(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("exact_match").eq(False))

def how_many_with_good_length(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("target_length").eq(pl.col("predicted_length")))

def get_last_token(s: str) -> str:
    return s.split(" | ")[-2]

def how_many_with_same_last_token(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("target").map_elements(
            get_last_token,
            pl.String
        ).alias("target_last_token"),
        pl.col("predicted").map_elements(
            get_last_token,
            pl.String
        ).alias("predicted_last_token"),
    )

    return df.filter(pl.col("target_last_token").eq(pl.col("predicted_last_token")))

def main(
        df_path: Path,
        output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_ndjson(df_path)
    # print(df.head())
    nb_test = df.height
    print(f"Number of tests: {nb_test:_}")

    df = keep_errors_only(df)

    good_length = how_many_with_good_length(df)
    same_last_token = how_many_with_same_last_token(df)

    nb_err = df.height
    print()
    print(f"Number of errors: {nb_err} ({nb_err / nb_test * 100:.3f} %) of the tests")
    print(f"Number of errors with good length: {good_length.height} ({good_length.height / nb_err * 100:.3f} %) of the errors, ({good_length.height / nb_test * 100:.3f} %) of the tests")
    print(f"Number of errors with same last token: {same_last_token.height} ({same_last_token.height / nb_err * 100:.3f} %) of the errors, ({same_last_token.height / nb_test * 100:.3f} %) of the tests")

    good_length_nb = good_length.height + (nb_test - nb_err)
    same_last_token_nb = same_last_token.height + (nb_test - nb_err)
    
    print()
    print(f"Number of any prediction with good length: {good_length_nb} ({good_length_nb / nb_test * 100:.3f} %) of the tests")
    print(f"Number of any prediction with same last token: {same_last_token_nb} ({same_last_token_nb / nb_test * 100:.3f} %) of the tests")

    df.write_ndjson(output_dir / "errors_only.jsonl")
    df.write_parquet(output_dir / "errors_only.parquet")
    good_length.write_ndjson(output_dir / "good_length.jsonl")
    good_length.write_parquet(output_dir / "good_length.parquet")
    same_last_token.write_ndjson(output_dir / "same_last_token.jsonl")
    same_last_token.write_parquet(output_dir / "same_last_token.parquet")

if __name__ == '__main__':
    main(
        df_path=Path("results_pho.ndjson"),
        output_dir=Path("errors_pho"),
    )
