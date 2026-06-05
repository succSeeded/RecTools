# SASRec vs UniSRec-ID Comparison

**Date:** 2026-05-18 10:14  
**GPU:** NVIDIA GeForce RTX 3090  
**Dataset:** ML-20M (min_rating=-1, min_item=5, min_user=2)

## Data

| | Count |
|---|---:|
| Interactions | 19,984,024 |
| Users | 138,493 |
| Items | 18,345 |
| Train | 19,707,038 |
| Val | 138,493 |
| Test | 138,493 |

## Config

| Parameter | Value |
|---|---|
| n_factors | 256 |
| n_blocks | 2 |
| n_heads | 1 |
| session_max_len | 200 |
| batch_size | 128 |
| lr | 0.001 |
| loss | softmax |
| optimizer | Adam |
| epochs | 10 |
| patience | None |
| dropout | 0.1 |

## Timing

| Stage | SASRec | UniSRec ID |
|---|---:|---:|
| Data load & split | 0.0s | 0.0s |
| Preprocessing | 6.7s | 0.1s |
| Model init | 0.0s | 0.0s |
| Training (10 epochs) | 1347.7s | 869.6s |
| Evaluation | 56.6s | 31.1s |
| **Total** | **1411.1s** | **900.8s** |

| | SASRec | UniSRec ID |
|---|---:|---:|
| Epochs completed | 11 | 10 |
| Time per epoch | 122.5s | 87.0s |
| Preprocessing speedup | — | 61x |

## Quality (test set, 138,493 users)

| Model | HR@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| SASRec | 0.2404 | 0.1398 | 0.1092 |
| UniSRec ID | 0.1856 | 0.1055 | 0.0811 |

UniSRec vs SASRec: HR@10 -22.8%, NDCG@10 -24.6%
