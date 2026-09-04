#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gguf>=0.10,<1",
#   "httpx>=0.27,<1",
#   "peft>=0.15,<1",
#   "safetensors>=0.4,<1",
#   "torch>=2.4,<2.15",
#   "transformers>=4.57,<6",
# ]
# ///
"""Compile a natural-language specification into a finetuned PAW program."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TEACHER_SYSTEM = (
    "You synthesize tiny supervised datasets for finetuning small language "
    "models from a natural-language task specification. Return strict JSON "
    "with a top-level key 'examples'. Each example must be an object with "
    "string keys 'input' and 'output'."
)
TEACHER_USER = (
    "Generate {n} diverse (input, output) example pairs that follow this "
    "specification.\n\n[SPEC]\n{spec}\n[END_SPEC]"
)
TEACHER_TEMPLATE_VERSION = "T1S4-v1-2026-05-10"

DEFAULT_API_URL = "https://programasweights.com"
DEFAULT_TEACHER_API_URL = "https://api.openai.com/v1"
DEFAULT_HF_ASSET_URL = "https://huggingface.co/programasweights/paw-programs/resolve/main"
DEFAULT_INTERPRETER = "Qwen/Qwen3-0.6B"
DEFAULT_BASE_COMPILER = "paw-4b-qwen3-0.6b"
DEFAULT_RECIPE_MEMORY_GIB = 38.0
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
MODULE_TO_GGUF = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
    "gate_proj": "ffn_gate",
    "up_proj": "ffn_up",
    "down_proj": "ffn_down",
}
MODULE_TO_PEFT = {
    "q_proj": "model.layers.{}.self_attn.q_proj",
    "k_proj": "model.layers.{}.self_attn.k_proj",
    "v_proj": "model.layers.{}.self_attn.v_proj",
    "o_proj": "model.layers.{}.self_attn.o_proj",
    "gate_proj": "model.layers.{}.mlp.gate_proj",
    "up_proj": "model.layers.{}.mlp.up_proj",
    "down_proj": "model.layers.{}.mlp.down_proj",
}

RUNTIME_MANIFEST = {
    "runtime_id": "qwen3-0.6b-q6_k",
    "manifest_version": 1,
    "display_name": "Qwen3 0.6B (Q6_K)",
    "interpreter": DEFAULT_INTERPRETER,
    "inference_provider_url": "http://localhost:9000",
    "adapter_format": "gguf_lora",
    "prompt_template": {
        "format": "rendered_text",
        "placeholder": "{INPUT_PLACEHOLDER}",
    },
    "program_assets": {
        "adapter_filename": "adapter.gguf",
        "prefix_cache_required": False,
        "prefix_cache_filename": None,
        "prefix_tokens_filename": None,
    },
    "local_sdk": {
        "supported": True,
        "base_model": {
            "provider": "huggingface",
            "repo": "programasweights/Qwen3-0.6B-GGUF-Q6_K",
            "file": "qwen3-0.6b-q6_k.gguf",
            "url": "https://huggingface.co/programasweights/Qwen3-0.6B-GGUF-Q6_K/resolve/main/qwen3-0.6b-q6_k.gguf",
            "size_bytes": 622733120,
            "sha256": "9a16ed5cacba959e63b62e2b6840c3eca2b51c3c3e51d31367ef8e4aafeae33c",
        },
        "n_ctx": 2048,
    },
    "js_sdk": {
        "supported": False,
        "base_model": None,
        "prefix_cache_supported": False,
    },
    "capabilities": {"python_local": True, "js_browser": False},
    "base_inference": {
        "contract_version": 1,
        "format": "rendered_text",
        "placeholder": "{INPUT_PLACEHOLDER}",
        "template": (
            "<|im_start|>user\n{INPUT_PLACEHOLDER}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
    },
}


@dataclass(frozen=True)
class Teacher:
    model: str
    examples: int
    temperature: float = 1.0
    examples_per_call: int = 8
    seed: int = 0


@dataclass
class Recipe:
    teachers: list[Teacher] = field(
        default_factory=lambda: [
            Teacher("gpt-5.4-mini", 2400),
            Teacher("gpt-5.5", 1200),
        ]
    )
    target_examples: int = 4800
    steps: int = 100
    batch_size: int = 48
    micro_batch_size: int = 48
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    warmup: float = 0.0
    lora_rank: int = 64
    lora_alpha: float = 16.0
    lora_init: str = "mapper"
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    max_length: int = 2048
    loss_chunk_size: int = 128
    synth_concurrency: int = 64
    retries: int = 3
    seed: int = 0
    base_compiler: str = DEFAULT_BASE_COMPILER
    interpreter: str = DEFAULT_INTERPRETER
    gradient_checkpointing: bool = False


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_teacher(value: str, defaults: Recipe) -> Teacher:
    try:
        model, raw_count = value.rsplit("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("teacher must be MODEL=EXAMPLES") from exc
    if not model or count <= 0:
        raise argparse.ArgumentTypeError("teacher must be MODEL=EXAMPLES with EXAMPLES > 0")
    return Teacher(
        model=model,
        examples=count,
        temperature=1.0,
        examples_per_call=defaults.teachers[0].examples_per_call,
        seed=defaults.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    defaults = Recipe()
    parser = argparse.ArgumentParser(
        description="Turn a natural-language specification into a reusable .paw program.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("spec", nargs="?", help="Natural-language function specification")
    parser.add_argument("--spec-file", type=Path, help="Read the specification from a file")
    parser.add_argument("-o", "--output", type=Path, default=Path("program.paw"))
    parser.add_argument(
        "--teacher",
        action="append",
        metavar="MODEL=EXAMPLES",
        help="Teacher and unique example count; repeat to mix teachers",
    )
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--examples-per-call", type=int, default=8)
    parser.add_argument("--target-examples", type=int, default=defaults.target_examples)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--micro-batch-size", type=int, default=defaults.micro_batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate)
    parser.add_argument("--warmup", type=float, default=defaults.warmup)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora-alpha", type=float, default=defaults.lora_alpha)
    parser.add_argument("--lora-init", choices=("mapper", "random"), default=defaults.lora_init)
    parser.add_argument(
        "--target-modules",
        default=",".join(defaults.target_modules),
        help="Comma-separated LoRA target modules",
    )
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--loss-chunk-size", type=int, default=defaults.loss_chunk_size)
    parser.add_argument("--synth-concurrency", type=int, default=defaults.synth_concurrency)
    parser.add_argument("--retries", type=int, default=defaults.retries)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--base-compiler", default=defaults.base_compiler)
    parser.add_argument("--interpreter", default=defaults.interpreter)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            os.environ.get("PAW_COMPILE_CACHE", "~/.cache/programasweights/compile-by-training")
        ).expanduser(),
    )
    parser.add_argument("--save-examples", type=Path)
    parser.add_argument("--keep-adapter", type=Path)
    parser.add_argument("--asset-timeout", type=int, default=180)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the final result as JSON")
    return parser


def recipe_from_args(args: argparse.Namespace) -> Recipe:
    defaults = Recipe()
    if args.teacher:
        teachers = [parse_teacher(value, defaults) for value in args.teacher]
    else:
        teachers = defaults.teachers
    teachers = [
        Teacher(
            teacher.model,
            teacher.examples,
            args.teacher_temperature,
            args.examples_per_call,
            args.seed,
        )
        for teacher in teachers
    ]
    target_modules = tuple(x.strip() for x in args.target_modules.split(",") if x.strip())
    recipe = Recipe(
        teachers=teachers,
        target_examples=args.target_examples,
        steps=args.steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup=args.warmup,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_init=args.lora_init,
        target_modules=target_modules,
        max_length=args.max_length,
        loss_chunk_size=args.loss_chunk_size,
        synth_concurrency=args.synth_concurrency,
        retries=args.retries,
        seed=args.seed,
        base_compiler=args.base_compiler,
        interpreter=args.interpreter,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    validate_recipe(recipe)
    return recipe


def validate_recipe(recipe: Recipe) -> None:
    positive_ints = {
        "target_examples": recipe.target_examples,
        "steps": recipe.steps,
        "batch_size": recipe.batch_size,
        "micro_batch_size": recipe.micro_batch_size,
        "lora_rank": recipe.lora_rank,
        "max_length": recipe.max_length,
        "loss_chunk_size": recipe.loss_chunk_size,
        "synth_concurrency": recipe.synth_concurrency,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if recipe.batch_size % recipe.micro_batch_size:
        raise ValueError("micro_batch_size must divide batch_size")
    if not 0.0 <= recipe.warmup < 1.0:
        raise ValueError("warmup must be in [0, 1)")
    if recipe.learning_rate <= 0 or recipe.min_learning_rate < 0:
        raise ValueError("learning rates must be non-negative and learning_rate must be positive")
    if not recipe.target_modules:
        raise ValueError("target_modules cannot be empty")
    unknown_modules = set(recipe.target_modules) - set(MODULE_TO_GGUF)
    if unknown_modules:
        raise ValueError(f"unsupported Qwen3 target modules: {sorted(unknown_modules)}")
    if recipe.interpreter != DEFAULT_INTERPRETER:
        raise ValueError(
            f"this script currently packages {DEFAULT_INTERPRETER}; got {recipe.interpreter}"
        )


def serializable_recipe(recipe: Recipe) -> dict[str, Any]:
    result = asdict(recipe)
    result["target_modules"] = list(recipe.target_modules)
    result["teacher_template"] = TEACHER_TEMPLATE_VERSION
    result["public_mapper_compile"] = True
    return result


def read_spec(args: argparse.Namespace) -> str:
    if args.spec and args.spec_file:
        raise ValueError("pass either a positional specification or --spec-file, not both")
    if args.spec_file:
        spec = args.spec_file.read_text(encoding="utf-8").strip()
    else:
        spec = (args.spec or "").strip()
    if len(spec) < 10:
        raise ValueError("the specification must contain at least 10 characters")
    return spec


def validate_destinations(args: argparse.Namespace) -> None:
    destinations = (
        ("output", args.output),
        ("saved examples", args.save_examples),
        ("kept adapter", args.keep_adapter),
    )
    for label, path in destinations:
        if path is not None and path.exists() and not args.force:
            raise FileExistsError(f"{label} path {path} already exists; pass --force to replace it")


def auth_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "User-Agent": "compile-by-training/1"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def compile_mapper(
    spec: str, recipe: Recipe, cache_dir: Path, asset_timeout: int
) -> dict[str, Any]:
    import httpx

    api_url = os.environ.get("PAW_API_URL", DEFAULT_API_URL).rstrip("/")
    paw_key = os.environ.get("PAW_API_KEY")
    log(f"1/4  Initializing with {recipe.base_compiler}...")
    response = httpx.post(
        f"{api_url}/api/v1/compile",
        json={
            "spec": spec,
            "compiler": recipe.base_compiler,
            "public": True,
            "ephemeral": False,
        },
        headers=auth_headers(paw_key),
        timeout=300.0,
    )
    response.raise_for_status()
    body = response.json()
    program_id = body.get("program_id")
    if not isinstance(program_id, str) or len(program_id) != 20:
        raise RuntimeError(f"PAW compiler returned an invalid program ID: {program_id!r}")
    pseudo_program = body.get("pseudo_program")
    if not isinstance(pseudo_program, str):
        raise RuntimeError("PAW compiler returned no pseudo-program")

    mapper_dir = cache_dir / "mapper" / program_id
    mapper_dir.mkdir(parents=True, exist_ok=True)
    download_mapper_asset(
        api_url,
        paw_key,
        program_id,
        "adapter_config.json",
        mapper_dir,
        asset_timeout,
    )
    download_mapper_asset(
        api_url,
        paw_key,
        program_id,
        "adapter_model.safetensors",
        mapper_dir,
        asset_timeout,
    )
    return {
        "program_id": program_id,
        "compiler_snapshot": body.get("compiler_snapshot") or recipe.base_compiler,
        "pseudo_program": pseudo_program,
        "pseudo_program_strategy": body.get("pseudo_program_strategy") or "vllm_generate",
        "adapter_dir": mapper_dir,
    }


def download_mapper_asset(
    api_url: str,
    api_key: str | None,
    program_id: str,
    filename: str,
    destination: Path,
    timeout: int,
) -> Path:
    import httpx

    output = destination / filename
    if output.is_file() and output.stat().st_size:
        return output
    hf_base = os.environ.get("PAW_HF_ASSET_URL", DEFAULT_HF_ASSET_URL).rstrip("/")
    urls = (
        f"{api_url}/api/v1/programs/{program_id}/asset/{filename}",
        f"{hf_base}/{program_id}/{filename}",
    )
    deadline = time.monotonic() + timeout
    last_status: dict[str, int | None] = {url: None for url in urls}
    while time.monotonic() < deadline:
        for url in urls:
            headers = auth_headers(api_key) if url.startswith(api_url) else None
            with httpx.stream(
                "GET",
                url,
                headers=headers,
                follow_redirects=True,
                timeout=120.0,
            ) as response:
                last_status[url] = response.status_code
                if response.status_code == 200:
                    temporary = output.with_suffix(output.suffix + ".part")
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_bytes(1024 * 1024):
                            handle.write(chunk)
                    if temporary.stat().st_size == 0:
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError(f"downloaded empty asset: {filename}")
                    temporary.replace(output)
                    return output
        time.sleep(3)
    statuses = ", ".join(str(status) for status in last_status.values())
    raise RuntimeError(f"timed out waiting for {filename} (HTTP {statuses})")


def parse_examples(content: Any) -> list[dict[str, str]]:
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return []
    else:
        value = content
    if isinstance(value, dict):
        value = next(
            (
                value.get(key)
                for key in ("examples", "data", "items")
                if isinstance(value.get(key), list)
            ),
            [],
        )
    if not isinstance(value, list):
        return []
    return [
        {"input": item["input"], "output": item["output"]}
        for item in value
        if isinstance(item, dict)
        and isinstance(item.get("input"), str)
        and isinstance(item.get("output"), str)
    ]


async def one_teacher_call(
    client: Any,
    semaphore: asyncio.Semaphore,
    spec: str,
    teacher: Teacher,
    seed_offset: int,
    api_key: str,
    api_url: str,
    retries: int,
) -> list[dict[str, str]]:
    payload = {
        "model": teacher.model,
        "temperature": teacher.temperature,
        "seed": teacher.seed + seed_offset,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {
                "role": "user",
                "content": TEACHER_USER.format(n=teacher.examples_per_call, spec=spec),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await client.post(
                    f"{api_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=240.0,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                examples = parse_examples(content)
                if len(examples) >= teacher.examples_per_call:
                    return examples[: teacher.examples_per_call]
            except Exception:
                if attempt == retries:
                    raise
            if attempt < retries:
                await asyncio.sleep(2**attempt)
    return []


def synth_cache_path(cache_dir: Path, spec: str, teacher: Teacher) -> Path:
    payload = {
        "spec": spec,
        "teacher": asdict(teacher),
        "template": TEACHER_TEMPLATE_VERSION,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cache_dir / "synth" / f"{digest}.json"


def cached_teacher_examples(
    cache_dir: Path, spec: str, teacher: Teacher
) -> list[dict[str, str]] | None:
    cache_path = synth_cache_path(cache_dir, spec, teacher)
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, list) or len(cached) < teacher.examples:
        return None
    examples = parse_examples(cached)
    if len(examples) < teacher.examples:
        return None
    return examples[: teacher.examples]


def validate_teacher_access(spec: str, recipe: Recipe, cache_dir: Path) -> bool:
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("TEACHER_API_KEY"):
        return False
    missing = [
        teacher.model
        for teacher in recipe.teachers
        if cached_teacher_examples(cache_dir, spec, teacher) is None
    ]
    if missing:
        raise RuntimeError(
            "set OPENAI_API_KEY (or TEACHER_API_KEY) for teacher synthesis; "
            f"no complete cache for {', '.join(missing)}"
        )
    return True


async def synthesize_teacher(
    client: Any,
    spec: str,
    teacher: Teacher,
    recipe: Recipe,
    api_key: str,
    api_url: str,
    cache_dir: Path,
) -> list[dict[str, str]]:
    cache_path = synth_cache_path(cache_dir, spec, teacher)
    cached = cached_teacher_examples(cache_dir, spec, teacher)
    if cached is not None:
        log(f"     {teacher.model}: using {teacher.examples} cached examples")
        return cached

    semaphore = asyncio.Semaphore(recipe.synth_concurrency)
    required_calls = math.ceil(teacher.examples / teacher.examples_per_call)
    next_seed = 0

    def new_task() -> asyncio.Task:
        nonlocal next_seed
        task = asyncio.create_task(
            one_teacher_call(
                client,
                semaphore,
                spec,
                teacher,
                next_seed,
                api_key,
                api_url,
                recipe.retries,
            )
        )
        next_seed += 1
        return task

    pending = {new_task() for _ in range(required_calls)}
    examples: list[dict[str, str]] = []
    replacements = 0
    max_replacements = max(16, required_calls)
    while pending and len(examples) < teacher.examples:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = await task
            if len(result) < teacher.examples_per_call:
                if replacements >= max_replacements:
                    continue
                pending.add(new_task())
                replacements += 1
                continue
            examples.extend(result)
            if len(examples) % 400 == 0 or len(examples) >= teacher.examples:
                count = min(len(examples), teacher.examples)
                log(f"     {teacher.model}: {count}/{teacher.examples}")
            if len(examples) >= teacher.examples:
                break
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if len(examples) < teacher.examples:
        raise RuntimeError(
            f"{teacher.model} produced only {len(examples)}/{teacher.examples} examples"
        )
    examples = examples[: teacher.examples]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{os.getpid()}.part")
    temporary.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return examples


async def synthesize(spec: str, recipe: Recipe, cache_dir: Path) -> list[list[dict[str, str]]]:
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TEACHER_API_KEY") or ""
    validate_teacher_access(spec, recipe, cache_dir)
    api_url = os.environ.get("TEACHER_API_URL", DEFAULT_TEACHER_API_URL).rstrip("/")
    log("2/4  Synthesizing training examples...")
    limits = httpx.Limits(
        max_connections=max(32, recipe.synth_concurrency * len(recipe.teachers) * 2),
        max_keepalive_connections=max(16, recipe.synth_concurrency * len(recipe.teachers)),
    )
    async with httpx.AsyncClient(limits=limits, timeout=240.0) as client:
        return await asyncio.gather(
            *[
                synthesize_teacher(client, spec, teacher, recipe, api_key, api_url, cache_dir)
                for teacher in recipe.teachers
            ]
        )


def render_prompt_template(pseudo_program: str, tokenizer: Any) -> str:
    pseudo = pseudo_program.strip()
    raw = (
        f"{pseudo}\n\n[INPUT]\n{{INPUT_PLACEHOLDER}}\n[END_INPUT]"
        if pseudo
        else "[INPUT]\n{INPUT_PLACEHOLDER}\n[END_INPUT]"
    )
    messages = [{"role": "user", "content": raw}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def tokenize_example(
    tokenizer: Any, prompt_template: str, item: dict[str, str], max_length: int
) -> dict[str, list[int]]:
    prompt = prompt_template.replace("{INPUT_PLACEHOLDER}", item["input"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    output_ids = tokenizer.encode(item["output"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        output_ids.append(tokenizer.eos_token_id)
    input_ids = (prompt_ids + output_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + output_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def allocate_quotas(counts: list[int], target: int) -> list[int]:
    total = sum(counts)
    exact = [target * count / total for count in counts]
    quotas = [math.floor(value) for value in exact]
    for index in sorted(range(len(counts)), key=lambda i: exact[i] - quotas[i], reverse=True)[
        : target - sum(quotas)
    ]:
        quotas[index] += 1
    return quotas


def build_schedule(
    pools: list[list[dict[str, list[int]]]], recipe: Recipe
) -> list[dict[str, list[int]]]:
    quotas = allocate_quotas(
        [teacher.examples for teacher in recipe.teachers], recipe.target_examples
    )
    required = recipe.steps * recipe.batch_size
    indexed_schedule: list[tuple[int, int]] = []
    epochs = math.ceil(required / recipe.target_examples)
    for epoch in range(epochs):
        one_epoch: list[tuple[int, int]] = []
        for component, (pool, quota) in enumerate(zip(pools, quotas, strict=True)):
            one_epoch.extend((component, index % len(pool)) for index in range(quota))
        random.Random(recipe.seed + epoch).shuffle(one_epoch)
        indexed_schedule.extend(one_epoch)
    if len(indexed_schedule) < required:
        raise RuntimeError("training schedule is shorter than steps * batch_size")

    # Production's "relabel" schedule preserves the shuffled repetition
    # pattern while assigning synthesized examples in completion order.
    mappings: list[dict[int, int]] = [{} for _ in pools]
    schedule: list[dict[str, list[int]]] = []
    for component, original_slot in indexed_schedule[:required]:
        mapping = mappings[component]
        if original_slot not in mapping:
            mapping[original_slot] = len(mapping)
        schedule.append(pools[component][mapping[original_slot]])
    return schedule


def collate(items: list[dict[str, list[int]]], pad_id: int, device: Any) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in items)
    input_ids, labels, attention = [], [], []
    for item in items:
        length = len(item["input_ids"])
        padding = max_len - length
        input_ids.append(item["input_ids"] + [pad_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        attention.append([1] * length + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


def unwrap_encoder_and_head(model: Any) -> tuple[Any, Any]:
    inner = getattr(getattr(model, "base_model", model), "model", None)
    if inner is None:
        raise RuntimeError("could not locate the PEFT base model")
    encoder = getattr(inner, "model", None) or getattr(inner, "transformer", None)
    head = getattr(inner, "lm_head", None)
    if encoder is None or head is None:
        raise RuntimeError(f"unsupported model layout: {type(inner).__name__}")
    return encoder, head


def chunked_loss_backward(
    model: Any,
    batch: dict[str, Any],
    loss_chunk_size: int,
    total_output_tokens: int,
) -> float:
    import torch
    import torch.nn.functional as functional

    encoder, head = unwrap_encoder_and_head(model)
    labels = batch["labels"]
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    hidden = encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).last_hidden_state
    detached = hidden.detach().requires_grad_(True)
    loss_sum = 0.0
    for start in range(0, detached.size(1), loss_chunk_size):
        end = min(start + loss_chunk_size, detached.size(1))
        labels_chunk = shifted[:, start:end]
        if not (labels_chunk != -100).any():
            continue
        logits = head(detached[:, start:end, :]).float()
        cross_entropy = functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels_chunk.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        (cross_entropy / total_output_tokens).backward()
        loss_sum += float(cross_entropy.detach())
        del logits, cross_entropy
    hidden.backward(detached.grad)
    return loss_sum


def cuda_is_advertised(torch: Any) -> bool:
    try:
        if torch.cuda.device_count() > 0:
            return True
    except Exception:
        pass
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SLURM_JOB_GPUS"):
        value = os.environ.get(name)
        if value is not None and value.strip().lower() not in ("", "-1", "none", "void"):
            return True
    return False


def choose_device(requested: str) -> Any:
    import torch

    if requested != "auto":
        device = torch.device(requested)
        if device.type not in ("cuda", "mps", "cpu"):
            raise ValueError("device must be auto, cuda, mps, or cpu")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if cuda_is_advertised(torch):
        raise RuntimeError(
            "a CUDA GPU is visible but PyTorch cannot initialize CUDA; "
            "check the driver/runtime instead of falling back to CPU"
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preflight_device(requested: str, recipe: Recipe) -> Any:
    import torch

    device = choose_device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot initialize CUDA")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    try:
        probe = torch.empty(1, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        del probe
    except Exception as exc:
        raise RuntimeError(f"{device} preflight allocation failed: {exc}") from exc

    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
        name = torch.cuda.get_device_name(device)
        log(f"Preflight: {device} ({name}; {free_gib:.1f}/{total_gib:.1f} GiB free)")
        if (
            free_gib < DEFAULT_RECIPE_MEMORY_GIB
            and recipe.micro_batch_size >= Recipe().micro_batch_size
            and not recipe.gradient_checkpointing
        ):
            log(
                "Warning: the default recipe uses about 38 GiB; reduce "
                "--micro-batch-size or enable --gradient-checkpointing."
            )
    elif device.type == "mps":
        log("Preflight: mps")
    else:
        log("Preflight: cpu")
        log("Warning: the default recipe is slow on CPU; use an accelerator when available.")
    return device


def cosine_learning_rate(step: int, total: int, high: float, low: float, warmup: float) -> float:
    warmup_steps = max(1, int(total * warmup))
    if step < warmup_steps:
        return high * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return low + 0.5 * (high - low) * (1.0 + math.cos(math.pi * progress))


def train(
    recipe: Recipe,
    mapper_dir: Path,
    pseudo_program: str,
    raw_pools: list[list[dict[str, str]]],
    output_dir: Path,
    device: Any,
) -> tuple[Path, str, list[float]]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("3/4  Finetuning the neural program...")
    random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(recipe.seed)
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    log(f"     device: {device}; dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(recipe.interpreter, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_template = render_prompt_template(pseudo_program, tokenizer)
    token_pools = [
        [tokenize_example(tokenizer, prompt_template, item, recipe.max_length) for item in pool]
        for pool in raw_pools
    ]
    schedule = build_schedule(token_pools, recipe)

    model = AutoModelForCausalLM.from_pretrained(
        recipe.interpreter,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    if recipe.lora_init == "mapper":
        mapper_config = json.loads((mapper_dir / "adapter_config.json").read_text(encoding="utf-8"))
        mapper_rank = int(mapper_config.get("r", 0))
        mapper_alpha = float(mapper_config.get("lora_alpha", 0))
        mapper_modules = set(mapper_config.get("target_modules") or [])
        if mapper_rank != recipe.lora_rank or mapper_alpha != recipe.lora_alpha:
            raise ValueError(
                f"mapper initialization has rank={mapper_rank}, alpha={mapper_alpha}; "
                "use those values or pass --lora-init random"
            )
        if mapper_modules != set(recipe.target_modules):
            raise ValueError(
                "mapper initialization has different target modules; "
                "use the defaults or pass --lora-init random"
            )
        peft_model = PeftModel.from_pretrained(model, str(mapper_dir), is_trainable=True)
    else:
        config = LoraConfig(
            r=recipe.lora_rank,
            lora_alpha=recipe.lora_alpha,
            target_modules=list(recipe.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, config)
    if recipe.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        peft_model.enable_input_require_grads()
    peft_model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in peft_model.parameters() if parameter.requires_grad),
        lr=recipe.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    losses: list[float] = []
    for step in range(recipe.steps):
        start = step * recipe.batch_size
        step_items = schedule[start : start + recipe.batch_size]
        total_output_tokens = sum(
            sum(label != -100 for label in item["labels"]) for item in step_items
        )
        if total_output_tokens == 0:
            raise RuntimeError("training batch contains no output tokens")
        learning_rate = cosine_learning_rate(
            step,
            recipe.steps,
            recipe.learning_rate,
            recipe.min_learning_rate,
            recipe.warmup,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro_start in range(0, recipe.batch_size, recipe.micro_batch_size):
            micro_items = step_items[micro_start : micro_start + recipe.micro_batch_size]
            batch = collate(micro_items, tokenizer.pad_token_id, device)
            loss_sum += chunked_loss_backward(
                peft_model,
                batch,
                recipe.loss_chunk_size,
                total_output_tokens,
            )
            del batch
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        loss = loss_sum / total_output_tokens
        losses.append(loss)
        log(f"     step {step + 1:>3}/{recipe.steps}  loss={loss:.4f}  lr={learning_rate:.2e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    del optimizer, peft_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_dir, prompt_template, losses


def peft_lora_parameters(adapter_dir: Path) -> dict[tuple[int, str], tuple[Any, Any]]:
    import re

    from safetensors.torch import load_file

    raw = load_file(str(adapter_dir / "adapter_model.safetensors"))
    patterns = {
        module: re.compile("^" + re.escape(template).replace(r"\{\}", r"(\d+)") + r"\.lora_(A|B)$")
        for module, template in MODULE_TO_PEFT.items()
    }
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for name, tensor in raw.items():
        clean = name
        for prefix in ("base_model.model.", "base_model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        if clean.endswith(".weight"):
            clean = clean[:-7]
        for module, pattern in patterns.items():
            match = pattern.fullmatch(clean)
            if match:
                grouped.setdefault((int(match.group(1)), module), {})[match.group(2)] = tensor.cpu()
                break
    result = {
        key: (pair["A"], pair["B"]) for key, pair in grouped.items() if "A" in pair and "B" in pair
    }
    if not result:
        raise RuntimeError("no supported LoRA tensors found in the trained adapter")
    return result


def convert_to_gguf(adapter_dir: Path, output: Path, alpha: float) -> Path:
    import gguf
    from gguf.quants import quantize

    parameters = peft_lora_parameters(adapter_dir)
    writer = gguf.GGUFWriter(str(output), arch="qwen3")
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", alpha)
    quantization = gguf.GGMLQuantizationType.Q4_0
    for (layer, module), (matrix_a, matrix_b) in parameters.items():
        base = f"blk.{layer}.{MODULE_TO_GGUF[module]}.weight"
        writer.add_tensor(
            f"{base}.lora_a",
            quantize(matrix_a.float().numpy(), quantization),
            raw_dtype=quantization,
        )
        writer.add_tensor(
            f"{base}.lora_b",
            quantize(matrix_b.float().numpy(), quantization),
            raw_dtype=quantization,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_program(
    spec: str,
    recipe: Recipe,
    mapper: dict[str, Any],
    adapter_dir: Path,
    prompt_template: str,
    losses: list[float],
    output: Path,
    force: bool,
) -> tuple[str, Path]:
    log("4/4  Packaging the .paw program...")
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compile-by-training-package-") as temporary:
        stage = Path(temporary)
        gguf_path = convert_to_gguf(adapter_dir, stage / "adapter.gguf", recipe.lora_alpha)
        recipe_json = serializable_recipe(recipe)
        fingerprint = hashlib.sha256(json.dumps(recipe_json, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        identity = {
            "spec": spec,
            "pseudo_program": mapper["pseudo_program"],
            "recipe": recipe_json,
            "adapter_sha256": sha256_file(gguf_path),
        }
        program_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        runtime = dict(RUNTIME_MANIFEST)
        meta = {
            "version": 4,
            "runtime_manifest_version": 1,
            "program_id": program_id,
            "spec": spec,
            "compiler_snapshot": "compile-by-training-local",
            "compiler_fingerprint": fingerprint,
            "compiler_kind": "finetune_lora",
            "interpreter": recipe.interpreter,
            "runtime_id": runtime["runtime_id"],
            "pseudo_program_strategy": mapper["pseudo_program_strategy"],
            "lora_rank": recipe.lora_rank,
            "lora_alpha": recipe.lora_alpha,
            "prefix_steps": 0,
            "runtime": runtime,
            "adapter": {
                "format": "gguf_lora",
                "lora_rank": recipe.lora_rank,
                "lora_alpha": recipe.lora_alpha,
                "target_modules": sorted(recipe.target_modules),
            },
            "program_assets": runtime["program_assets"],
            "prompt_template": runtime["prompt_template"],
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_program_id": mapper["program_id"],
            "base_compiler_snapshot": mapper["compiler_snapshot"],
            "compile_by_training": {
                "recipe": recipe_json,
                "final_loss": losses[-1],
                "adapter_sha256": identity["adapter_sha256"],
            },
            "examples": [],
        }
        meta_text = json.dumps(meta, indent=2, ensure_ascii=False)
        (stage / "meta.json").write_text(meta_text, encoding="utf-8")
        (stage / "pseudo_program.txt").write_text(mapper["pseudo_program"], encoding="utf-8")
        (stage / "prompt_template.txt").write_text(prompt_template, encoding="utf-8")
        bundle = stage / f"{program_id}.paw"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as archive:
            for filename in (
                "meta.json",
                "pseudo_program.txt",
                "adapter.gguf",
                "prompt_template.txt",
            ):
                archive.write(stage / filename, filename)
        temporary_output = output.with_suffix(output.suffix + ".part")
        shutil.copy2(bundle, temporary_output)
        temporary_output.replace(output)
        install_in_sdk_cache(program_id, bundle, stage)
    return program_id, output


def install_in_sdk_cache(program_id: str, bundle: Path, stage: Path) -> Path:
    cache_root = Path(os.environ.get("PAW_CACHE_DIR", "~/.cache/programasweights")).expanduser()
    programs = cache_root / "programs"
    destination = programs / program_id
    programs.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = programs / f".{program_id}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for filename in ("meta.json", "pseudo_program.txt", "adapter.gguf", "prompt_template.txt"):
            shutil.copy2(stage / filename, temporary / filename)
        shutil.copy2(bundle, temporary / f"{program_id}.paw")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def flatten_examples(recipe: Recipe, pools: Iterable[list[dict[str, str]]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for teacher, pool in zip(recipe.teachers, pools, strict=True):
        output.extend({"teacher": teacher.model, **example} for example in pool)
    return output


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        started = time.perf_counter()
        recipe = recipe_from_args(args)
        if args.print_config:
            print(json.dumps(serializable_recipe(recipe), indent=2))
            return 0
        spec = read_spec(args)
        if args.output.suffix != ".paw":
            args.output = args.output.with_suffix(args.output.suffix + ".paw")
        validate_destinations(args)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        if validate_teacher_access(spec, recipe, args.cache_dir):
            log("Preflight: all teacher examples are cached; no teacher API key needed.")
        device = preflight_device(args.device, recipe)
        mapper = compile_mapper(spec, recipe, args.cache_dir, args.asset_timeout)
        pools = asyncio.run(synthesize(spec, recipe, args.cache_dir))
        if args.save_examples:
            args.save_examples.parent.mkdir(parents=True, exist_ok=True)
            args.save_examples.write_text(
                json.dumps(flatten_examples(recipe, pools), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        with tempfile.TemporaryDirectory(prefix="compile-by-training-") as temporary:
            adapter_dir, prompt_template, losses = train(
                recipe,
                mapper["adapter_dir"],
                mapper["pseudo_program"],
                pools,
                Path(temporary) / "adapter",
                device,
            )
            if args.keep_adapter:
                if args.keep_adapter.exists():
                    if not args.force:
                        raise FileExistsError(
                            f"{args.keep_adapter} already exists; pass --force to replace it"
                        )
                    shutil.rmtree(args.keep_adapter)
                shutil.copytree(adapter_dir, args.keep_adapter)
            program_id, output = package_program(
                spec,
                recipe,
                mapper,
                adapter_dir,
                prompt_template,
                losses,
                args.output,
                args.force,
            )
        result = {
            "program_id": program_id,
            "path": str(output.resolve()),
            "base_program_id": mapper["program_id"],
            "final_loss": losses[-1],
            "device": str(device),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if args.json:
            print(json.dumps(result))
        else:
            print(f"\nCompiled {output}")
            print(f"Program ID: {program_id}")
            print("\nRun it locally:")
            print("  import programasweights as paw")
            print(f'  fn = paw.function("{program_id}")')
            print('  print(fn("your input"))')
        return 0
    except KeyboardInterrupt:
        log("Interrupted.")
        return 130
    except Exception as exc:
        log(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
