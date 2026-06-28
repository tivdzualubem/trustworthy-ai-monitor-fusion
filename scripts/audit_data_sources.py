from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


OUTPUT_PATH = Path("data/metadata/source_audit.json")


def git_head(url: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", url, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()[0]


def hf_dataset_metadata(repo_id: str) -> dict[str, Any]:
    info = HfApi().dataset_info(repo_id=repo_id)

    card_data = info.card_data
    license_value = None

    if card_data is not None:
        try:
            license_value = card_data.get("license")
        except (AttributeError, TypeError):
            license_value = None

    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "private": info.private,
        "gated": info.gated,
        "license": license_value,
        "last_modified": (
            info.last_modified.isoformat()
            if info.last_modified is not None
            else None
        ),
    }


def main() -> None:
    audit = {
        "audited_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "selected_dataset_count": 3,
        "sources": {
            "jailbreakbench": {
                "type": "huggingface_dataset",
                **hf_dataset_metadata(
                    "JailbreakBench/JBB-Behaviors"
                ),
                "configuration": "judge_comparison",
                "subset": "adversarial_indices_0_199",
                "held_out_attack_family": "GCG",
            },
            "xstest": {
                "type": "git",
                "url": (
                    "https://github.com/"
                    "paul-rottger/xstest.git"
                ),
                "revision": git_head(
                    "https://github.com/"
                    "paul-rottger/xstest.git"
                ),
                "selected_file": (
                    "model_completions/"
                    "xstest_v2_completions_gpt4.csv"
                ),
                "selected_subset": "safe_prompts_only",
                "license": "CC-BY-4.0",
            },
            "wildguardmix": {
                "type": "huggingface_dataset",
                **hf_dataset_metadata(
                    "allenai/wildguardmix"
                ),
                "configuration": "wildguardtest",
                "split": "test",
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
