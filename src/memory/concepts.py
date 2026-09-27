"""Which ML concepts a piece of text is about: a curated vocabulary with aliases.

Why a vocabulary instead of an LLM or keyphrase model: it is free, instant and deterministic, and
mastery needs *stable identities* - "LoRA", "low-rank adaptation" and "LoRA fine-tuning" must
update the same row. An open-ended extractor would invent a new spelling each time. The cost is
coverage: concepts outside the list are invisible to the memory until added here (ADR 0008).
"""

import re

# canonical slug -> aliases (matched case-insensitively on word boundaries)
VOCABULARY: dict[str, list[str]] = {
    # architectures & building blocks
    "transformer": ["transformer", "transformers"],
    "self_attention": ["self-attention", "self attention", "scaled dot-product attention"],
    "multi_head_attention": ["multi-head attention", "multihead attention", "mha"],
    "attention": ["attention mechanism", "attention"],
    "positional_encoding": ["positional encoding", "positional embedding", "position embedding"],
    "rope": ["rope", "rotary position embedding", "rotary embedding", "rotary positional"],
    "layer_norm": ["layer normalization", "layernorm", "layer norm", "rmsnorm"],
    "residual_connection": ["residual connection", "skip connection", "resnet"],
    "mixture_of_experts": ["mixture of experts", "mixture-of-experts", "moe"],
    "state_space_model": ["state space model", "state-space model", "mamba", "ssm"],
    "cnn": ["convolutional neural network", "cnn", "convolution"],
    "rnn": ["recurrent neural network", "rnn", "lstm", "gru"],
    "vision_transformer": ["vision transformer", "vit"],
    "diffusion_model": ["diffusion model", "diffusion models", "denoising diffusion", "ddpm"],
    "gan": ["generative adversarial network", "gan", "gans"],
    "vae": ["variational autoencoder", "vae"],
    "autoencoder": ["autoencoder"],
    "graph_neural_network": ["graph neural network", "gnn", "message passing"],
    "embedding": ["embedding", "embeddings", "word2vec"],
    "tokenization": ["tokenization", "tokenizer", "byte pair encoding", "bpe", "sentencepiece"],
    # training & optimisation
    "gradient_descent": ["gradient descent", "sgd", "stochastic gradient"],
    "backpropagation": ["backpropagation", "backprop"],
    "adam": ["adam", "adamw"],
    "learning_rate_schedule": ["learning rate schedule", "warmup", "cosine schedule"],
    "regularization": ["regularization", "weight decay", "l2 penalty"],
    "dropout": ["dropout"],
    "batch_norm": ["batch normalization", "batchnorm", "batch norm"],
    "overfitting": ["overfitting", "generalization gap"],
    "cross_entropy": ["cross-entropy", "cross entropy", "log loss"],
    "contrastive_learning": ["contrastive learning", "contrastive loss", "infonce", "simclr"],
    "self_supervised_learning": ["self-supervised", "self supervised"],
    "pretraining": ["pretraining", "pre-training"],
    "fine_tuning": ["fine-tuning", "fine tuning", "finetuning"],
    "transfer_learning": ["transfer learning"],
    "scaling_laws": ["scaling law", "scaling laws", "chinchilla"],
    "curriculum_learning": ["curriculum learning"],
    # efficient adaptation & inference
    "lora": ["lora", "low-rank adaptation", "low rank adaptation"],
    "qlora": ["qlora"],
    "peft": ["parameter-efficient fine-tuning", "peft", "adapters", "prefix tuning"],
    "quantization": ["quantization", "quantized", "int8", "4-bit", "nf4"],
    "distillation": ["knowledge distillation", "distillation"],
    "pruning": ["pruning", "sparsity"],
    "kv_cache": ["kv cache", "kv-cache", "key-value cache"],
    "flash_attention": ["flashattention", "flash attention"],
    "speculative_decoding": ["speculative decoding"],
    "long_context": ["long context", "context window", "context length"],
    # LLM behaviour & alignment
    "language_model": ["language model", "language models", "llm", "llms"],
    "in_context_learning": ["in-context learning", "in context learning", "few-shot prompting"],
    "chain_of_thought": ["chain-of-thought", "chain of thought", "cot"],
    "prompt_engineering": ["prompt engineering", "prompting"],
    "instruction_tuning": ["instruction tuning", "instruction-tuned", "supervised fine-tuning"],
    "rlhf": ["rlhf", "reinforcement learning from human feedback", "reward model"],
    "dpo": ["dpo", "direct preference optimization"],
    "alignment": ["alignment", "ai safety"],
    "hallucination": ["hallucination", "hallucinations"],
    "retrieval_augmented_generation": [
        "retrieval-augmented generation",
        "retrieval augmented",
        "rag",
    ],
    "agents": ["llm agent", "llm agents", "tool use", "function calling"],
    "decoding": [
        "beam search",
        "top-p",
        "nucleus sampling",
        "greedy decoding",
        "temperature sampling",
    ],
    # RL & probabilistic
    "reinforcement_learning": ["reinforcement learning", "policy gradient", "q-learning"],
    "ppo": ["ppo", "proximal policy optimization"],
    "bayesian_inference": ["bayesian", "posterior", "variational inference"],
    "kl_divergence": ["kl divergence", "kullback-leibler", "kl-divergence"],
    "maximum_likelihood": ["maximum likelihood", "mle", "likelihood"],
    # evaluation
    "evaluation_benchmarks": ["benchmark", "benchmarks", "mmlu", "glue"],
    "perplexity": ["perplexity"],
    "ablation_study": ["ablation", "ablations"],
    "statistical_significance": ["statistical significance", "confidence interval", "p-value"],
    # retrieval & representation
    "dense_retrieval": ["dense retrieval", "bi-encoder", "vector search"],
    "bm25": ["bm25", "sparse retrieval"],
    "reranking": ["reranking", "re-ranking", "cross-encoder"],
    "clip": ["clip", "contrastive language-image"],
    "multimodal": ["multimodal", "vision-language", "vlm"],
}

# Longest aliases first, so "multi-head attention" wins over "attention" at the same position.
_ALIASES = sorted(
    ((alias, slug) for slug, aliases in VOCABULARY.items() for alias in aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
# Optional plural suffix: "VAEs", "GANs", "transformers" match their singular alias. Without it,
# "What does the reparameterization trick in VAEs do?" found no concept and the domain gate
# refused it (caught by the offline routing eval).
_PATTERN = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(alias) for alias, _ in _ALIASES) + r")(?:e?s)?(?![\w-])",
    re.IGNORECASE,
)
_SLUG_OF = {alias.lower(): slug for alias, slug in _ALIASES}


def extract_concepts(text: str, limit: int = 6) -> list[str]:
    """Canonical concept slugs mentioned in `text`, in order of first mention, deduplicated."""
    found: list[str] = []
    for match in _PATTERN.finditer(text):
        slug = _SLUG_OF[match.group(1).lower()]
        if slug not in found:
            found.append(slug)
            if len(found) == limit:
                break
    return found


def is_known(concept: str) -> bool:
    return concept in VOCABULARY


# How a concept is written for people, where the first alias's casing is not it.
DISPLAY = {
    "lora": "LoRA",
    "qlora": "QLoRA",
    "rope": "RoPE",
    "rlhf": "RLHF",
    "dpo": "DPO",
    "ppo": "PPO",
    "peft": "Parameter-efficient fine-tuning",
    "gan": "GANs",
    "vae": "Variational autoencoders",
    "cnn": "Convolutional networks",
    "rnn": "Recurrent networks",
    "bm25": "BM25",
    "clip": "CLIP",
    "kv_cache": "KV cache",
    "flash_attention": "FlashAttention",
    "adam": "Adam",
    "layer_norm": "Layer normalization",
    "batch_norm": "Batch normalization",
    "language_model": "Language models",
    "multimodal": "Multimodal models",
    "retrieval_augmented_generation": "Retrieval-augmented generation",
    "kl_divergence": "KL divergence",
}


def label(concept: str) -> str:
    """Human-readable name for UIs ("lora" -> "LoRA", "self_attention" -> "Self-attention")."""
    if concept in DISPLAY:
        return DISPLAY[concept]
    first = VOCABULARY.get(concept, [concept.replace("_", " ")])[0]
    return first[:1].upper() + first[1:]
