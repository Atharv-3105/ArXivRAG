"""Run the whole product locally with no Docker, no database and no API keys.

    python -m src.api.demo            # http://localhost:8000

In-memory stores (users, profiles, history, graph), a small built-in corpus of well-known papers
(short paraphrases of their abstracts and methods, enough to exercise every screen), and the LLM
router from .env: with no keys that is the offline extractive mock, with keys the real providers.
Everything is lost when the process stops. For development and demos only - never deploy it.
"""

import asyncio
import os
import re

os.environ.setdefault("SECRET_KEY", "demo-mode-secret-key-not-for-production")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://demo:demo@localhost:5432/demo")

from src.storage.vector_db import SearchHit  # noqa: E402

# (arxiv_id, title, year, category, [(section, text)])
PAPERS = [
    (
        "1706.03762v7",
        "Attention Is All You Need",
        "2017-06-12",
        "cs.CL",
        [
            (
                "Abstract",
                "The Transformer is a sequence transduction model based solely on attention "
                "mechanisms, dispensing with recurrence and convolutions entirely. It is more "
                "parallelizable and requires significantly less time to train.",
            ),
            (
                "3.2 Attention",
                "Scaled dot-product attention computes "
                "$\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}(QK^\\top / \\sqrt{d_k})\\,V$. "
                "Dividing by $\\sqrt{d_k}$ keeps the dot products from growing large and pushing the "
                "softmax into regions with extremely small gradients.",
            ),
        ],
    ),
    (
        "1810.04805v2",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "2018-10-11",
        "cs.CL",
        [
            (
                "Abstract",
                "BERT pre-trains deep bidirectional representations from unlabeled text by "
                "jointly conditioning on both left and right context in all layers, using a masked "
                "language model objective.",
            ),
        ],
    ),
    (
        "2005.14165v4",
        "Language Models are Few-Shot Learners",
        "2020-05-28",
        "cs.CL",
        [
            (
                "Abstract",
                "GPT-3 is an autoregressive language model with 175 billion parameters. Tasks "
                "and few-shot demonstrations are specified purely via text interaction with the model, "
                "without any gradient updates or fine-tuning.",
            ),
        ],
    ),
    (
        "2106.09685v2",
        "LoRA: Low-Rank Adaptation of Large Language Models",
        "2021-06-17",
        "cs.CL",
        [
            (
                "Abstract",
                "LoRA freezes the pretrained model weights and injects trainable rank "
                "decomposition matrices into each layer of the Transformer, greatly reducing the number of "
                "trainable parameters for downstream tasks. Compared with GPT-3 175B fine-tuned with Adam, "
                "LoRA can reduce the number of trainable parameters by 10,000 times.",
            ),
            (
                "4.1 Low-rank-parametrized update matrices",
                "For a pretrained weight matrix "
                "$W_0 \\in \\mathbb{R}^{d \\times k}$, LoRA constrains its update with a low-rank "
                "decomposition $W_0 + \\Delta W = W_0 + BA$, where $B \\in \\mathbb{R}^{d \\times r}$, "
                "$A \\in \\mathbb{R}^{r \\times k}$ and the rank $r \\ll \\min(d, k)$. During training "
                "$W_0$ is frozen while $A$ and $B$ are trained, so the forward pass becomes "
                "$h = W_0 x + BAx$.",
            ),
            (
                "4.2 No additional inference latency",
                "Because the product $BA$ can be merged into "
                "$W_0$ after training, LoRA introduces no additional inference latency compared with a "
                "fully fine-tuned model.",
            ),
        ],
    ),
    (
        "2305.14314v1",
        "QLoRA: Efficient Finetuning of Quantized LLMs",
        "2023-05-23",
        "cs.LG",
        [
            (
                "Abstract",
                "QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained "
                "language model into Low Rank Adapters. It introduces 4-bit NormalFloat, double "
                "quantization and paged optimizers, and makes it possible to finetune a 65B parameter "
                "model on a single 48GB GPU.",
            ),
        ],
    ),
    (
        "2205.14135v2",
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "2022-05-27",
        "cs.LG",
        [
            (
                "Abstract",
                "FlashAttention is an IO-aware exact attention algorithm that uses tiling to "
                "reduce the number of memory reads and writes between GPU high bandwidth memory and "
                "on-chip SRAM.",
            ),
        ],
    ),
    (
        "2203.02155v1",
        "Training language models to follow instructions with human feedback",
        "2022-03-04",
        "cs.CL",
        [
            (
                "Abstract",
                "Language models are fine-tuned with human feedback: supervised fine-tuning on "
                "labeler demonstrations, then reinforcement learning from human feedback using a reward "
                "model trained on human rankings of model outputs, optimized with PPO.",
            ),
        ],
    ),
    (
        "2305.18290v3",
        "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "2023-05-29",
        "cs.LG",
        [
            (
                "Abstract",
                "Direct Preference Optimization fits the policy directly to human preference "
                "data with a simple classification loss, without fitting an explicit reward model or "
                "running reinforcement learning during fine-tuning.",
            ),
        ],
    ),
    (
        "1707.06347v2",
        "Proximal Policy Optimization Algorithms",
        "2017-07-20",
        "cs.LG",
        [
            (
                "Abstract",
                "Proximal Policy Optimization alternates between sampling data through "
                "interaction with the environment and optimizing a clipped surrogate objective with "
                "several epochs of minibatch updates.",
            ),
        ],
    ),
    (
        "1412.6980v9",
        "Adam: A Method for Stochastic Optimization",
        "2014-12-22",
        "cs.LG",
        [
            (
                "Abstract",
                "Adam is an algorithm for first-order gradient-based optimization of "
                "stochastic objective functions, based on adaptive estimates of lower-order moments.",
            ),
        ],
    ),
]

# (citing, cited, function, citing sentence)
CITATIONS = [
    (
        "1810.04805v2",
        "1706.03762",
        "builds_on",
        "BERT's model architecture is a multi-layer bidirectional Transformer encoder.",
    ),
    (
        "2005.14165v4",
        "1706.03762",
        "builds_on",
        "GPT-3 uses the same model and architecture as GPT-2, a Transformer decoder.",
    ),
    (
        "2106.09685v2",
        "1706.03762",
        "neutral",
        "LoRA is applied to the weight matrices of the Transformer's self-attention module.",
    ),
    (
        "2106.09685v2",
        "2005.14165",
        "comparative",
        "LoRA performs on par with or better than fine-tuning GPT-3 175B on quality.",
    ),
    ("2106.09685v2", "1412.6980", "neutral", "The baseline fine-tunes GPT-3 175B with Adam."),
    (
        "2305.14314v1",
        "2106.09685",
        "builds_on",
        "QLoRA backpropagates through a frozen 4-bit model into LoRA adapters.",
    ),
    (
        "2205.14135v2",
        "1706.03762",
        "critical",
        "Transformers are slow and memory-hungry on long sequences, since self-attention is quadratic in sequence length.",
    ),
    (
        "2203.02155v1",
        "2005.14165",
        "builds_on",
        "The InstructGPT models are fine-tuned versions of GPT-3.",
    ),
    (
        "2203.02155v1",
        "1707.06347",
        "builds_on",
        "The policy is optimized against the reward model with PPO.",
    ),
    (
        "2305.18290v3",
        "2203.02155",
        "critical",
        "RLHF is a complex and often unstable procedure, fitting a reward model and then fine-tuning with RL.",
    ),
    (
        "2305.18290v3",
        "1707.06347",
        "comparative",
        "DPO matches or exceeds PPO-based RLHF without reinforcement learning.",
    ),
]

EQUATIONS = {
    "2106.09685v2": [
        {
            "latex": "h = W_0 x + BAx",
            "number": "(3)",
            "section_path": "4.1 Low-rank-parametrized update matrices",
        }
    ],
    "1706.03762v7": [
        {
            "latex": "\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V",
            "number": "(1)",
            "section_path": "3.2 Attention",
        }
    ],
}

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "are",
    "how",
    "what",
    "does",
    "why",
    "explain",
}


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class DemoRetriever:
    """Word-overlap ranking over the built-in corpus: enough to exercise the product offline."""

    def __init__(self) -> None:
        self.chunks: list[SearchHit] = []
        for pid, title, published, _cat, sections in PAPERS:
            for i, (section, text) in enumerate(sections):
                self.chunks.append(
                    SearchHit(
                        chunk_id=f"{pid}:{i}",
                        paper_id=pid,
                        title=title,
                        section=section,
                        content=text,
                        score=0.0,
                        metadata={
                            "math_fidelity": "latex",
                            "parse_method": "html",
                            "anchor": f"S{i}",
                        },
                        published=published,
                    )
                )

    async def search(self, query, top_k=None, paper_ids=None, mode="hybrid"):
        from src.memory.concepts import VOCABULARY, extract_concepts

        # "DPO" should find "Direct Preference Optimization": expand with the concepts' aliases
        # (a stand-in for what dense embeddings do in the real hybrid search).
        aliases = " ".join(a for c in extract_concepts(query) for a in VOCABULARY[c])
        q = _words(f"{query} {aliases}")
        scored = []
        for hit in self.chunks:
            if paper_ids and hit.paper_id not in paper_ids:
                continue
            overlap = len(q & _words(f"{hit.title} {hit.section} {hit.content}"))
            if overlap or paper_ids:
                scored.append((overlap, hit))
        scored.sort(key=lambda pair: -pair[0])
        return [SearchHit(**{**vars(h), "score": float(s)}) for s, h in scored[: top_k or 12]]


async def build_services():
    from src.api.services import Services
    from src.auth.users import InMemoryUserStore
    from src.core.metrics import observe_llm
    from src.graph.influence import refresh_influence
    from src.graph.store import InMemoryGraphStore
    from src.llm.registry import build_router
    from src.memory.store import InMemoryProfileStore
    from src.storage.conversations import InMemoryConversationStore
    from src.storage.paper_content import InMemoryPaperContent

    graph = InMemoryGraphStore()
    content = InMemoryPaperContent()
    for pid, title, published, category, sections in PAPERS:
        graph.add_paper(pid, title, published, category)
        content.add(
            pid,
            title,
            [
                {"title": s, "path": s, "anchor": f"S{i}", "chunks": [t]}
                for i, (s, t) in enumerate(sections)
            ],
            EQUATIONS.get(pid),
            published,
        )
    for citing, cited, function, sentence in CITATIONS:
        graph.cite(citing, cited, function, sentence)
    await refresh_influence(graph)
    return Services(
        llm=build_router(observer=observe_llm),
        retriever=DemoRetriever(),
        conversations=InMemoryConversationStore(),
        users=InMemoryUserStore(),
        profiles=InMemoryProfileStore(),
        graph_store=graph,
        content=content,
    )


def main() -> None:
    import uvicorn

    from src.api.app import create_app

    services = asyncio.run(build_services())
    print("ArXivMind demo: http://localhost:8000  (in-memory; data is lost on exit)")
    uvicorn.run(
        create_app(services=services), host="127.0.0.1", port=int(os.getenv("PORT", "8000"))
    )


if __name__ == "__main__":
    main()
