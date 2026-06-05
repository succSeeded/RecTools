"""Compare RecTools SASRecModel and RecTools.semantic TIGER model on a dataset that comes with rich item metadata --
Amazon-2018 Toys & Games. Both models are trained for 10 epochs with batch size of 256.

Usage:
```bash
python3 -m benchmark.sasrec_vs_tiger
```

Data is doenloaded automatically if not present. Item embeddings are generated using
sentence-transformers/sentence-t5-base model.
"""

# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
# pylint: disable=import-error,unsubscriptable-object,protected-access,cyclic-import

import argparse
import gc
import gzip
import json
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm, trange

from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import MRR, NDCG, CatalogCoverage, HitRate
from rectools.models import SASRecModel
from rectools.models.nn.item_net import IdEmbeddingsItemNet
from rectools.models.nn.transformers.utils import leave_one_out_mask
from rectools.semantic.data_handling import loo_split
from rectools.semantic.metrics import gini_k
from rectools.semantic.tiger import TIGERModel
from rectools.semantic.tokenizer import SIDTokenizer

DATASET_URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Toys_and_Games_5.json.gz"
META_URLS = (
    "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Toys_and_Games.json.gz",
    "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles/meta_Toys_and_Games.json.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SASRec from rectools.models against semantic TIGER on Amazon Toys and Games 5-core."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("benchmark/data"),
        help="Directory for downloaded data and cached artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override. Defaults to cuda -> mps -> cpu.",
    )
    parser.add_argument(
        "--top-k-main",
        type=int,
        default=10,
        help="K for NDCG, MRR, coverage, and gini.",
    )
    parser.add_argument("--max-length", type=int, default=20, help="Maximum context length")
    parser.add_argument("--top-k-hit", type=int, default=10, help="K for hit rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Optional cap on users after preprocessing.",
    )
    parser.add_argument(
        "--tokenizer-max-epochs",
        type=int,
        default=1000,
        help="Tokenizer training epochs.",
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=2048,
        help="Tokenizer training epochs.",
    )
    parser.add_argument(
        "--tokenizer-patience",
        type=int,
        default=100,
        help="Tokenizer early stopping patience.",
    )
    parser.add_argument("--sasrec-epochs", type=int, default=10, help="Maximum SASRec epochs.")
    parser.add_argument(
        "--sasrec-patience",
        type=int,
        default=10,
        help="SASRec early stopping patience.",
    )
    parser.add_argument("--sasrec-batch-size", type=int, default=256, help="SASRec batch size.")
    parser.add_argument("--sasrec-num-workers", type=int, default=4, help="SASRec dataloader workers.")
    parser.add_argument("--tiger-epochs", type=int, default=10, help="Maximum TIGER epochs.")
    parser.add_argument("--tiger-patience", type=int, default=10, help="TIGER early stopping patience.")
    parser.add_argument("--tiger-batch-size", type=int, default=256, help="TIGER train batch size.")
    parser.add_argument("--tiger-eval-batch-size", type=int, default=256, help="TIGER eval batch size.")
    parser.add_argument("--tiger-num-workers", type=int, default=4, help="TIGER dataloader workers.")
    return parser.parse_args()


def resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_index = torch.device(device).index or 0
        return f"{device} ({torch.cuda.get_device_name(gpu_index)})"
    if device == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def maybe_download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst}")
    urlretrieve(url, dst)  # nosec B310


def download_meta(meta_path: Path) -> None:
    if meta_path.exists():
        return
    last_error: Optional[Exception] = None
    for url in META_URLS:
        try:
            maybe_download(url, meta_path)
            return
        except (HTTPError, URLError) as exc:
            last_error = exc
    raise RuntimeError(f"Failed to download metadata from any known URL: {last_error}")


def read_gz_json_lines(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_interactions(dataset_path: Path, limit_users: Optional[int]) -> pd.DataFrame:
    rows = []
    for _, record in enumerate(tqdm(read_gz_json_lines(dataset_path), desc="Loading interactions")):
        rows.append(
            {
                "user_id": record["reviewerID"],
                "item_id": record["asin"],
                "timestamp": pd.to_datetime(record["unixReviewTime"], unit="s"),
            }
        )

    interactions = pd.DataFrame(rows).sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    if limit_users is not None:
        kept_users = interactions["user_id"].drop_duplicates().iloc[:limit_users]
        interactions = interactions[interactions["user_id"].isin(kept_users)].reset_index(drop=True)

    return interactions


def fill_missing_meta(meta: pd.DataFrame, item_ids: set) -> pd.DataFrame:
    """Add meta entries with text 'Unknown' for items present in interactions but missing from meta."""
    missing_items = sorted(item_ids - set(meta["item_id"]))
    if missing_items:
        missing_df = pd.DataFrame({"item_id": missing_items, "text": "Unknown"})
        meta = pd.concat([meta, missing_df], ignore_index=True)
    return meta


def build_meta_file(meta_gz_path: Path, interactions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_csv(out_path)

    item_ids = set(interactions["item_id"].unique())
    rows = []
    for record in read_gz_json_lines(meta_gz_path):
        item_id = record.get("asin")
        if item_id not in item_ids:
            continue

        title = record.get("title") or record.get("name") or ""
        description = record.get("description") or ""
        if isinstance(description, list):
            description = " ".join(str(part) for part in description)
        title = str(title).strip()
        description = str(description).strip()

        text = f"{title}: {description}"
        rows.append({"item_id": item_id, "text": text})

    meta = pd.DataFrame(rows).drop_duplicates("item_id")
    meta = meta[meta["item_id"].isin(interactions["item_id"].unique())].copy()
    meta["text"] = meta["text"].fillna("Unknown").astype(str)

    meta = fill_missing_meta(meta, item_ids)
    meta = meta.sort_values("item_id").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_path, index=False)
    return meta


def map_ids(interactions: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_codes = pd.Index(interactions["user_id"].unique())
    item_codes = pd.Index(interactions["item_id"].unique())

    user_map = pd.Series(np.arange(len(user_codes), dtype=np.int64), index=user_codes)
    item_map = pd.Series(np.arange(len(item_codes), dtype=np.int64), index=item_codes)

    mapped_interactions = interactions.copy()
    mapped_interactions["user_id"] = mapped_interactions["user_id"].map(user_map).astype(np.int64)
    mapped_interactions["item_id"] = mapped_interactions["item_id"].map(item_map).astype(np.int64)

    mapped_meta = meta[meta["item_id"].isin(item_map.index)].copy()
    mapped_meta["item_id"] = mapped_meta["item_id"].map(item_map).astype(np.int64)
    mapped_meta = mapped_meta.sort_values("item_id").reset_index(drop=True)
    return mapped_interactions, mapped_meta


def encode_texts(texts: list, model_name: str, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Encode texts into embeddings using a SentenceTransformer model.

    Parameters
    ----------
    texts : list of str
        Texts to encode.
    model_name : str
        Name or path of a sentence-transformers model.
    device : str
        Device to run the model on.
    batch_size : int
        Batch size for encoding.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape ``(len(texts), embed_dim)``.
    """
    embeds = []
    model = SentenceTransformer(model_name).to(device).eval()
    with torch.no_grad():
        for start in trange(0, len(texts), batch_size, desc="Embedding the metadata"):
            end = min(start + batch_size, len(texts))
            embeds.append(model.encode(texts[start:end]))

    return np.vstack(embeds)


def make_sasrec_dataset(val_df: pd.DataFrame) -> Dataset:
    sasrec_df = val_df.rename(columns={"timestamp": Columns.Datetime}).copy()
    sasrec_df[Columns.Weight] = 1.0
    return Dataset.construct(interactions_df=sasrec_df[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]])


def get_last_item_targets(test_df: pd.DataFrame) -> pd.DataFrame:
    targets = test_df.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False).tail(1).copy()
    targets["weight"] = 1.0
    return targets[["user_id", "item_id", "weight"]]


def reco_to_topk_lists(reco: pd.DataFrame, k: int) -> list[list[int]]:
    ordered = reco[reco[Columns.Rank] <= k].sort_values([Columns.User, Columns.Rank])
    grouped = ordered.groupby(Columns.User, sort=False)[Columns.Item].agg(list)
    return grouped.tolist()


def evaluate_recommendations(
    reco: pd.DataFrame,
    targets: pd.DataFrame,
    catalog_size: int,
    top_k_main: int,
    top_k_hit: int,
) -> dict[str, float]:
    metrics = {
        f"NDCG@{top_k_main}": NDCG(k=top_k_main).calc(reco, targets),
        f"HR@{top_k_hit}": HitRate(k=top_k_hit).calc(reco, targets),
        f"MRR@{top_k_main}": MRR(k=top_k_main).calc(reco, targets),
        f"Coverage@{top_k_main}": CatalogCoverage(k=top_k_main, normalize=True).calc(
            reco, catalog=list(range(catalog_size))
        ),
        f"Gini@{top_k_main}": gini_k(reco_to_topk_lists(reco, top_k_main), top_k_main),
    }
    return metrics


def train_sasrec(
    dataset: Dataset,
    device: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    max_length: int,
) -> SASRecModel:
    trainer = Trainer(
        accelerator="gpu" if device.startswith("cuda") else device,
        devices=1,
        min_epochs=1,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(
                monitor=SASRecModel.val_loss_name,
                patience=patience,
                mode="min",
            )
        ],
        deterministic=True,
        precision="bf16-mixed",
    )

    model = SASRecModel(
        n_factors=200,
        n_blocks=1,
        n_heads=2,
        dropout_rate=0.1,
        train_min_user_interactions=2,
        session_max_len=max_length,
        item_net_block_types=(IdEmbeddingsItemNet,),
        get_val_mask_func=leave_one_out_mask,
        batch_size=batch_size,
        dataloader_num_workers=num_workers,
        verbose=1,
        deterministic=True,
        recommend_torch_device=device,
    )
    model._trainer = trainer
    model.fit(dataset)
    return model


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_sasrec(
    val_df: pd.DataFrame,
    test_targets: pd.DataFrame,
    device: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float, float]:
    print("Converting DataFrame to RecTools dataset")
    dataset = make_sasrec_dataset(val_df)

    print("Fitting RecTools SASRec")
    fit_start = time.perf_counter()
    model = train_sasrec(
        dataset=dataset,
        device=device,
        max_epochs=args.sasrec_epochs,
        patience=args.sasrec_patience,
        batch_size=args.sasrec_batch_size,
        num_workers=args.sasrec_num_workers,
        max_length=args.max_length,
    )
    train_seconds = time.perf_counter() - fit_start

    users = test_targets["user_id"].to_numpy()
    inference_start = time.perf_counter()
    reco = model.recommend(
        users=users,
        dataset=dataset,
        k=max(args.top_k_main, args.top_k_hit),
        filter_viewed=False,
    )
    inference_seconds = time.perf_counter() - inference_start

    metrics = evaluate_recommendations(
        reco=reco,
        targets=test_targets,
        catalog_size=val_df["item_id"].nunique(),
        top_k_main=args.top_k_main,
        top_k_hit=args.top_k_hit,
    )
    return metrics, train_seconds, inference_seconds


def benchmark_tiger(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_targets: pd.DataFrame,
    meta: pd.DataFrame,
    device: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float, float]:
    meta = meta.sort_values("item_id").drop_duplicates("item_id")
    embeddings = encode_texts(
        meta["text"].tolist(),
        model_name="sentence-transformers/sentence-t5-base",
        device=device,
        batch_size=args.tokenizer_batch_size,
    )
    tokenizer = SIDTokenizer(
        input_dim=embeddings.shape[1],
        codebook_sizes=[256, 256, 256],
        codebook_dim=32,
        hidden_dims=[768, 256, 128, 64],
        quantizer="rqvae",
        device=device,
    )
    tokenizer.fit(
        item_ids=meta["item_id"].tolist(),
        embeddings=embeddings,
        max_epochs=args.tokenizer_max_epochs,
        patience=args.tokenizer_patience,
        batch_size=2048,
        save_dir=str(args.workdir / "tokenizer_checkpoints"),
    )

    tiger = TIGERModel(
        tokenizer=tokenizer,
        hidden_units=128,
        num_blocks=4,
        num_heads=4,
        dropout_rate=0.1,
        max_length=args.max_length,
        lr=1e-3,
        lr_schedule="cosine",
        max_epochs=args.tiger_epochs,
        patience=args.tiger_patience,
        batch_size=args.tiger_batch_size,
        eval_batch_size=args.tiger_eval_batch_size,
        num_workers=args.tiger_num_workers,
        beam_size=max(20, args.top_k_hit),
        top_k=args.top_k_main,
        d_kv=64,
        device=device,
    )

    fit_start = time.perf_counter()
    tiger.fit(train_df=train_df, val_df=val_df)
    train_seconds = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    reco = tiger.predict(val_df, top_k=max(args.top_k_main, args.top_k_hit))
    inference_seconds = time.perf_counter() - inference_start

    metrics = evaluate_recommendations(
        reco=reco,
        targets=test_targets,
        catalog_size=val_df["item_id"].nunique(),
        top_k_main=args.top_k_main,
        top_k_hit=args.top_k_hit,
    )
    return metrics, train_seconds, inference_seconds


def print_results(results: dict[str, dict[str, float]], device: str) -> None:
    print()
    print("Benchmark device:", describe_device(device))
    print()

    metric_names = [
        "train_seconds",
        "inference_seconds",
        "NDCG@10",
        "HR@10",
        "MRR@10",
        "Coverage@10",
        "Gini@10",
    ]

    header = f"{'model':<10}" + "".join(f"{name:>18}" for name in metric_names)
    print(header)
    print("-" * len(header))
    for model_name, model_metrics in results.items():
        row = f"{model_name:<10}" + "".join(f"{model_metrics[name]:>18.6f}" for name in metric_names)
        print(row)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    device = resolve_device(args.device)
    args.workdir.mkdir(parents=True, exist_ok=True)

    dataset_path = args.workdir / "Toys_and_Games_5.json.gz"
    meta_gz_path = args.workdir / "meta_Toys_and_Games.json.gz"
    meta_csv_path = args.workdir / "meta_toys_and_games_5.csv"

    maybe_download(DATASET_URL, dataset_path)
    download_meta(meta_gz_path)

    interactions = load_interactions(dataset_path, limit_users=args.limit_users)
    meta = build_meta_file(meta_gz_path, interactions, meta_csv_path)
    interactions, meta = map_ids(interactions, meta)

    train_df, val_df, test_df = loo_split(interactions, timestamp_col="timestamp")
    test_targets = get_last_item_targets(test_df)

    print(
        "Prepared dataset:",
        f"{len(interactions):,} interactions,",
        f"{interactions['user_id'].nunique():,} users,",
        f"{interactions['item_id'].nunique():,} items",
    )

    sasrec_metrics, sasrec_train_s, sasrec_infer_s = benchmark_sasrec(val_df, test_targets, device, args)
    cleanup()
    tiger_metrics, tiger_train_s, tiger_infer_s = benchmark_tiger(
        train_df=train_df,
        val_df=val_df,
        test_targets=test_targets,
        meta=meta,
        device=device,
        args=args,
    )

    results = {
        "SASRec": {
            "train_seconds": sasrec_train_s,
            "inference_seconds": sasrec_infer_s,
            **sasrec_metrics,
        },
        "TIGER": {
            "train_seconds": tiger_train_s,
            "inference_seconds": tiger_infer_s,
            **tiger_metrics,
        },
    }
    print_results(results, device)


if __name__ == "__main__":
    main()
