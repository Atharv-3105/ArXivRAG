"""The evaluation corpus: well-known arXiv papers the retrieval labels refer to.

    python -m src.eval.seed        # ingest them (python -m src.pipeline.runner --ids ...)

A fixed id list, not a search query, so every run evaluates against the same corpus.
"""

import asyncio

SEED_IDS = [
    "2106.09685",  # LoRA
    "2305.14314",  # QLoRA
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3
    "2305.18290",  # DPO
    "2203.02155",  # InstructGPT
    "2205.14135",  # FlashAttention
    "2104.09864",  # RoFormer (RoPE)
    "2203.15556",  # Chinchilla
    "2010.11929",  # ViT
    "2103.00020",  # CLIP
    "2006.11239",  # DDPM
    "2312.00752",  # Mamba
    "2101.03961",  # Switch Transformer
    "2201.11903",  # Chain-of-thought
    "2005.11401",  # RAG
    "1707.06347",  # PPO
    "1412.6980",  # Adam
    "1502.03167",  # Batch normalization
    "1512.03385",  # ResNet
    "1607.06450",  # Layer normalization
    "1301.3781",  # word2vec
    "1406.2661",  # GANs
    "1312.6114",  # VAE
    "1503.02531",  # Knowledge distillation
    "2001.08361",  # Scaling laws
    "2211.17192",  # Speculative decoding
    "2309.06180",  # vLLM / PagedAttention
]


def main() -> None:
    from src.core.logging import setup_logging
    from src.pipeline.runner import run_pipeline

    setup_logging()
    asyncio.run(run_pipeline(ids=SEED_IDS, max_papers=len(SEED_IDS)))


if __name__ == "__main__":
    main()
