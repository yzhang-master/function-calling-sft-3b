import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from function_calling_sft import default_registry
from function_calling_sft.trajectories import Message, trajectory_from_dict


def to_chat_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.function_call is not None:
            converted.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": message.function_call.name,
                                "arguments": json.dumps(
                                    message.function_call.arguments, sort_keys=True
                                ),
                            },
                        }
                    ],
                }
            )
        else:
            converted.append({"role": message.role, "content": message.content})
            if message.name is not None:
                converted[-1]["name"] = message.name
    return converted


class TrajectoryDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
        self.examples = []
        tools = [tool.json_schema() for tool in default_registry().all()]
        for line in path.read_text().splitlines():
            trajectory = trajectory_from_dict(json.loads(line))
            text = tokenizer.apply_chat_template(
                to_chat_messages(trajectory.messages),
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            self.examples.append(
                {"input_ids": encoded["input_ids"], "labels": encoded["input_ids"].copy()}
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        length = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((len(features), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(features), length), dtype=torch.long)
        for row, feature in enumerate(features):
            size = len(feature["input_ids"])
            input_ids[row, :size] = torch.tensor(feature["input_ids"])
            labels[row, :size] = torch.tensor(feature["labels"])
            attention_mask[row, :size] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-only causal SFT for a configurable 3B model")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_3b.json"))
    parser.add_argument("--output", type=Path, default=Path("results/qwen25-3b-sft"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("The 3B entry point requires a CUDA GPU; use train_tiny.py on CPU")
    if config["training"].get("bf16") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured 3B run requires BF16-capable CUDA hardware")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        message = 'Install the optional training dependencies: pip install -e ".[sft]"'
        raise RuntimeError(message) from error

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = TrajectoryDataset(Path(config["data_path"]), tokenizer, config["max_length"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output),
        overwrite_output_dir=True,
        report_to=[],
        save_strategy="steps",
        **config["training"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    print(
        {
            "model": config["model_name_or_path"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trajectories": len(dataset),
            "device": torch.cuda.get_device_name(0),
        }
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))


if __name__ == "__main__":
    main()
