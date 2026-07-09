from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/metadata/monitor_registry.json"
AUDIT_PATH = ROOT / "data/metadata/model_access_audit.json"

SMALL_METADATA_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "chat_template.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_model(repo_id: str) -> dict[str, Any]:
    api = HfApi()
    record: dict[str, Any] = {
        "repo_id": repo_id,
        "checked_at": utc_now(),
        "access_ok": False,
        "revision": None,
        "gated": None,
        "private": None,
        "small_files": {},
        "error": None,
    }

    try:
        info = api.model_info(repo_id=repo_id)
        record["revision"] = info.sha
        record["gated"] = info.gated
        record["private"] = info.private
        record["access_ok"] = True

        for filename in SMALL_METADATA_FILES:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=info.sha,
                    repo_type="model",
                )
                size_bytes = Path(path).stat().st_size
                record["small_files"][filename] = {
                    "status": "downloaded_metadata_only",
                    "size_bytes": size_bytes,
                }
            except EntryNotFoundError:
                record["small_files"][filename] = {
                    "status": "not_present"
                }

    except GatedRepoError as exc:
        record["error"] = (
            "Gated repository. Accept the model terms on Hugging Face "
            "and rerun this script."
        )
        record["exception_type"] = type(exc).__name__

    except RepositoryNotFoundError as exc:
        record["error"] = (
            "Repository not found or not accessible with the current token."
        )
        record["exception_type"] = type(exc).__name__

    except Exception as exc:
        record["error"] = str(exc)
        record["exception_type"] = type(exc).__name__

    return record


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    model_records = []
    by_repo: dict[str, dict[str, Any]] = {}

    for monitor in registry["monitors"]:
        repo_id = monitor.get("model_repo")
        if repo_id is None:
            monitor["model_access_required"] = False
            monitor["model_revision"] = None
            continue

        result = verify_model(repo_id)
        model_records.append(result)
        by_repo[repo_id] = result

        monitor["model_access_required"] = True
        monitor["model_access_verified"] = bool(result["access_ok"])
        monitor["model_revision"] = result["revision"]
        monitor["model_revision_pinned_at"] = (
            result["checked_at"] if result["access_ok"] else None
        )

    audit = {
        "audit_version": "0.1.0",
        "created_at": utc_now(),
        "purpose": (
            "Verify Hugging Face access and pin immutable revisions for "
            "learned monitors without downloading model weights."
        ),
        "download_policy": {
            "full_model_weights_downloaded": False,
            "metadata_files_only": SMALL_METADATA_FILES,
            "note": (
                "The script uses hf_hub_download only for named config and "
                "tokenizer metadata files. It does not request weight files."
            ),
        },
        "models": model_records,
    }

    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== MODEL ACCESS RESULTS ===")
    failed = False

    for result in model_records:
        print()
        print("Repo:", result["repo_id"])
        print("Access OK:", result["access_ok"])
        print("Revision:", result["revision"])
        print("Gated:", result["gated"])
        print("Private:", result["private"])

        if result["error"]:
            failed = True
            print("Error:", result["error"])
        else:
            downloaded = [
                name
                for name, meta in result["small_files"].items()
                if meta["status"] == "downloaded_metadata_only"
            ]
            print("Metadata files downloaded:", downloaded)

    print()
    print("Updated:", REGISTRY_PATH)
    print("Wrote:", AUDIT_PATH)

    if failed:
        raise SystemExit(
            "One or more model repositories could not be verified."
        )


if __name__ == "__main__":
    main()
