# function-calling-sft-3b

> **Era:** 2023–2024 · Senior AI Engineer focus area (agents). Technique from 2023, reimplemented in 2026.

Synthetic multi-turn typed-tool trajectories, strict schema/argument validation, and actual gradient fine-tuning of a small PyTorch function caller. A configurable full-model 3B SFT entry point is included, but **no 3B model was trained on this CPU machine and no 3B accuracy is claimed.**

## Problem

Function-calling data is often reduced to isolated prompt/JSON pairs, skipping tool responses and follow-up turns. A training pipeline also needs to reject unknown tools, missing or extra arguments, wrong JSON types, invalid enum values, and malformed calls. Reported exact-match gains should identify the model and data distribution rather than implying that a tiny proxy measures a production LLM.

## Method

### Typed tools and multi-turn data

The local registry defines `get_weather`, `schedule_meeting`, and `set_reminder`. Every argument has a type; enums and numeric bounds are enforced, and all listed parameters are required. `json_schema()` emits function schemas with `additionalProperties: false`. `parse_function_call()` accepts only a JSON object with exactly `name` and `arguments`, then runs registry validation.

The seeded generator creates deterministic eight-message trajectories:

```text
system → user → assistant function_call → tool result
       → user follow-up → assistant function_call → tool result → assistant final
```

Each trajectory has two schema-validated calls. Weather/scheduling and reminder/weather orders alternate. JSONL serialization preserves calls and tool-response names; loading revalidates the complete trajectory. The committed corpus contains 256 training trajectories (512 calls) and 64 test trajectories (128 calls).

### Trainable tiny proxy

For each assistant call, `extract_call_examples()` uses the most recent user turn as model input and aligns every argument value to a source-token span. Missing or conflicting spans fail generation. This yields supervised examples from the actual multi-turn trajectories, not hard-coded predictions at inference.

`TinyToolCaller` is a 32-dimensional token embedding followed by a 32-hidden-unit bidirectional GRU. A pooled head classifies the tool and a per-token head predicts `O` or one of eight argument slots. Training minimizes the sum of tool and token cross-entropy, with non-`O` tags weighted 2×. Decoding joins selected tokens, casts numeric slots, and must pass the same schema validator. Exact accuracy requires both tool name and the complete argument dictionary to match.

The untrained comparison is the same randomly initialized architecture before optimization. It is **not a pretrained base LLM**.

## Results

Actual CPU proxy run:

```bash
PYTHONPATH=src python scripts/train_tiny.py
```

Training used 160 trajectories / 320 calls; held-out evaluation used a different seed with 48 trajectories / 96 calls. The model has 15,564 parameters and a 66-token vocabulary. It trained for 80 epochs with batch size 128, learning rate 0.01, and seed 7. Full output and failure examples are in `results/tiny_proxy_metrics.json`.

| Model state (96 held-out calls) | Exact calls | Function-call accuracy | Tool accuracy | Schema-valid calls |
|---|---:|---:|---:|---:|
| Randomly initialized base proxy | 0 / 96 | 0.00% | 0.00% | 0.00% |
| Fine-tuned tiny proxy | 96 / 96 | 100.00% | 100.00% | 100.00% |

Mean batch training loss fell from 3.000262 to 0.000670. The independent smoke run trains on only eight calls and reports loss 3.1647 → 0.0029 and 100% training-set exact accuracy.

### 3B training: pending

| Model | Training | Base-vs-SFT accuracy |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` (default, configurable) | **Not run locally** | **Pending GPU run** |

`configs/sft_3b.json` configures full causal SFT with BF16, gradient checkpointing, two epochs, batch size 1, gradient accumulation 16, and learning rate 1e-5. `scripts/train_3b.py` converts trajectories to the selected tokenizer's tool-aware chat template, tokenizes them, pads labels with `-100`, refuses CPU-only execution, trains through Transformers `Trainer`, and saves the model and tokenizer. It currently applies loss to the full rendered sequence, not only assistant spans.

On a CUDA machine with a CUDA-enabled PyTorch build and optional dependencies installed, the exact training command is:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_3b.py --config configs/sft_3b.json --output results/qwen25-3b-sft
```

Full BF16 3B training with AdamW needs substantial memory for weights, gradients, optimizer states, and activations; an 80GB GPU is a reasonable starting point, but memory use has not been measured here. Change `model_name_or_path` to fine-tune a different causal model. No local 3B checkpoint, training loss, or accuracy is included.

## Reproduce

Python 3.10+ and CPU PyTorch are enough for all tests and tiny proxy results; tests use scripted local data and need no API keys.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest
PYTHONPATH=src python scripts/smoke.py
PYTHONPATH=src python scripts/generate_data.py
PYTHONPATH=src python scripts/train_tiny.py
```

For the unexecuted 3B workflow, use a CUDA PyTorch environment, run `pip install -e ".[sft]"`, then use the exact GPU command above. The shared local environment does not include Transformers, so only Python syntax—not model loading or GPU training—was checked for that entry point.

## Limitations

- The proxy test set uses the same templates and value pools as training with a different seed. Its 100% score demonstrates learning of this synthetic task, not open-domain generalization, robustness, or 3B capability.
- The base proxy is random and untrained; comparing it with the fitted proxy is not equivalent to comparing pretrained base and SFT LLMs.
- Slot tagging assumes argument values appear verbatim in the latest user turn. It cannot synthesize dates, resolve coreference, call multiple tools in parallel, or infer arguments from older context and tool results.
- The corpus has three tools and two calls per trajectory. Tool outputs are deterministic synthetic JSON, not live weather or calendar services.
- The 3B entry point has not been executed locally, and its full-sequence loss may spend capacity on user/tool tokens. Assistant-only masking, evaluation of generated 3B calls, distributed training, and memory profiling remain future work.

## References

- T. Schick et al. *Toolformer: Language Models Can Teach Themselves to Use Tools.* 2023. arXiv:2302.04761
- S. Patil et al. *Gorilla: Large Language Model Connected with Massive APIs.* 2023. arXiv:2305.15334
- Qwen Team. *Qwen2.5 Technical Report.* 2024. arXiv:2412.15115
- K. Cho et al. *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* 2014. arXiv:1406.1078

## License

MIT
