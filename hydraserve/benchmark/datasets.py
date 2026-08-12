"""
Benchmark dataset loaders and generators.

Supports five datasets for comprehensive evaluation:
  - ShareGPT: Real conversation traces (1K-8K context)
  - HumanEval: Code completion scenarios (2K-8K context)
  - LongBench: Long document QA (8K-128K context)
  - WikiText-103: Perplexity validation
  - GSM8K: Math reasoning accuracy

Also provides synthetic load generators for controlled experiments:
  - Fixed concurrent (closed-loop throughput test)
  - Burst arrival (TTFT distribution test)
  - Poisson arrival (realistic traffic simulation)
  - Mixed context (real-world distribution)
"""

from typing import List, Dict, Tuple, Optional, Iterator
import random
import math
import json


# ─── Dataset Loaders ────────────────────────────────────────────────

def load_sharegpt(path: str = None, max_samples: int = 1000) -> List[Dict]:
    """
    Load ShareGPT conversation dataset.

    Returns list of {"conversations": [...], "id": str}.
    """
    if path is None:
        return _generate_synthetic_conversations(max_samples)
    with open(path) as f:
        data = json.load(f)
    return data[:max_samples]


def load_humaneval(path: str = None) -> List[Dict]:
    """
    Load HumanEval code completion dataset.

    Returns list of {"task_id": str, "prompt": str, "canonical_solution": str, ...}.
    """
    if path is None:
        return _generate_synthetic_code_prompts(164)
    with open(path) as f:
        data = json.load(f)
    return data


def load_longbench(path: str = None) -> List[Dict]:
    """
    Load LongBench long-context dataset.

    Returns list of {"input": str, "context": str, "answers": [...], "length": str}.
    """
    if path is None:
        return _generate_synthetic_long_docs(100)
    with open(path) as f:
        data = json.load(f)
    return data


def load_wikitext(path: str = None, max_tokens: int = 131072) -> str:
    """
    Load WikiText-103 for perplexity evaluation.

    Returns concatenated text.
    """
    if path is None:
        return _generate_synthetic_text(max_tokens)
    with open(path) as f:
        return f.read()[:max_tokens * 4]  # ~4 chars per token


def load_gsm8k(path: str = None) -> List[Dict]:
    """
    Load GSM8K math reasoning dataset.

    Returns list of {"question": str, "answer": str}.
    """
    if path is None:
        return _generate_synthetic_math_problems(100)
    with open(path) as f:
        data = json.load(f)
    return data


# ─── Synthetic Generators ───────────────────────────────────────────

def _generate_synthetic_conversations(n: int) -> List[Dict]:
    """Generate synthetic conversation-like data."""
    topics = ["programming", "writing", "analysis", "math", "science", "history"]
    conversations = []
    for i in range(n):
        topic = random.choice(topics)
        num_turns = random.randint(2, 8)
        conv = []
        for _ in range(num_turns):
            conv.append({
                "from": "human" if _ % 2 == 0 else "gpt",
                "value": f"Synthetic {topic} conversation turn {_} " * random.randint(20, 200),
            })
        conversations.append({"conversations": conv, "id": f"synthetic_{i}"})
    return conversations


def _generate_synthetic_code_prompts(n: int) -> List[Dict]:
    """Generate synthetic code completion prompts."""
    prompts = []
    for i in range(n):
        prompts.append({
            "task_id": f"HumanEval/{i}",
            "prompt": f"def function_{i}(x):\n    \"\"\"Synthetic function {i}.\"\"\"\n    ",
            "canonical_solution": f"    return x * {i} + {i * 2}",
        })
    return prompts


def _generate_synthetic_long_docs(n: int) -> List[Dict]:
    """Generate synthetic long documents."""
    docs = []
    for i in range(n):
        context_len = random.choice([8000, 16000, 32000, 64000, 128000])
        docs.append({
            "input": f"Question {i}: What is discussed in this document?",
            "context": f"Document {i} content: " + ("word " * (context_len // 5)),
            "answers": [f"Answer to question {i}"],
            "length": f"{context_len // 1000}K",
        })
    return docs


def _generate_synthetic_text(n_tokens: int) -> str:
    """Generate synthetic text at ~4 chars per token."""
    words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
             "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"]
    return " ".join(random.choice(words) for _ in range(n_tokens))


def _generate_synthetic_math_problems(n: int) -> List[Dict]:
    """Generate synthetic math word problems."""
    problems = []
    for i in range(n):
        a, b = random.randint(1, 100), random.randint(1, 100)
        problems.append({
            "question": f"If Alice has {a} apples and Bob has {b} apples, "
                        f"how many apples do they have together?",
            "answer": f"{a + b}",
        })
    return problems


# ─── Load Generators ─────────────────────────────────────────────────


class LoadGenerator:
    """Base class for load generators."""

    def __init__(self, dataset: List[Dict], seed: int = 42):
        self.dataset = dataset
        self.rng = random.Random(seed)
        self.idx = 0

    def next(self) -> Tuple[str, int]:
        """Return (prompt_text, prompt_len) for next request."""
        raise NotImplementedError


class FixedConcurrencyGenerator(LoadGenerator):
    """
    Fixed concurrency (closed-loop) generator.

    Maintains N concurrent requests. When one completes, immediately
    starts another. Used for throughput testing.
    """

    def __init__(self, dataset: List[Dict], prompt_lengths: Optional[List[int]] = None,
                 seed: int = 42):
        super().__init__(dataset, seed)
        self.prompt_lengths = prompt_lengths or [1024, 2048, 4096, 8192]

    def next(self) -> Tuple[str, int]:
        target_len = self.rng.choice(self.prompt_lengths)
        return ("benchmark " * (target_len // 2), target_len)


class BurstArrivalGenerator(LoadGenerator):
    """
    Burst arrival generator.

    Simulates N requests arriving simultaneously (e.g., coding assistant
    with multiple developers sending context simultaneously).
    Used for TTFT distribution testing.
    """

    def __init__(self, dataset: List[Dict], burst_size: int = 5,
                 prompt_lengths: Optional[List[int]] = None, seed: int = 42):
        super().__init__(dataset, seed)
        self.burst_size = burst_size
        self.prompt_lengths = prompt_lengths or [4096, 8192, 16384, 32768]

    def generate_burst(self) -> List[Tuple[str, int]]:
        """Generate a burst of requests."""
        burst = []
        for _ in range(self.burst_size):
            target_len = self.rng.choice(self.prompt_lengths)
            burst.append(("burst " * (target_len // 2), target_len))
        return burst


class PoissonArrivalGenerator(LoadGenerator):
    """
    Poisson arrival generator.

    Requests arrive according to a Poisson process with configurable
    lambda (average requests per second). Simulates realistic traffic.
    Used for P50/P99 TPOT testing.
    """

    def __init__(self, dataset: List[Dict], lambda_rps: float = 5.0,
                 prompt_lengths: Optional[List[int]] = None, seed: int = 42):
        super().__init__(dataset, seed)
        self.lambda_rps = lambda_rps
        self.prompt_lengths = prompt_lengths or [1024, 4096, 8192, 16384]

    def next_arrival_delay(self) -> float:
        """Return delay until next request (exponential distribution)."""
        return self.rng.expovariate(self.lambda_rps)

    def next(self) -> Tuple[str, int]:
        target_len = self.rng.choice(self.prompt_lengths)
        return ("poisson " * (target_len // 2), target_len)


class MixedContextGenerator(LoadGenerator):
    """
    Mixed context generator (real-world distribution).

    Short prompts (1K-4K): 60%
    Medium prompts (8K-32K): 30%
    Long prompts (64K-128K): 10%

    Used for realistic workload simulation.
    """

    def __init__(self, dataset: List[Dict], seed: int = 42):
        super().__init__(dataset, seed)

    def next(self) -> Tuple[str, int]:
        r = self.rng.random()
        if r < 0.60:
            target_len = self.rng.randint(1000, 4000)
        elif r < 0.90:
            target_len = self.rng.randint(8000, 32000)
        else:
            target_len = self.rng.randint(64000, 128000)
        return ("mixed " * (target_len // 2), target_len)
