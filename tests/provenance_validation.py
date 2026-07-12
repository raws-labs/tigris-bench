"""Integrity and provenance checks for tracked benchmark result artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
RESULT_ARTIFACT_NAME = "summary.json"
UNKNOWN_EXECUTION_FIELDS = {
    "capture_timestamp_utc",
    "benchmark_repository_revision",
    "tigris_compiler_revision",
    "tigris_runtime_revision",
    "tflite_micro_revision",
    "arm_none_eabi_toolchain_version",
    "siliconrig_revision",
    "pico_sdk_revision",
    "build_flags",
    "model_artifact_hashes",
    "firmware_hashes",
    "board_identifiers",
}
EXPECTED_DEPENDENCIES = {
    "CMSIS-NN",
    "CMSIS-Core",
    "cmsis-device-f4",
    "cmsis-device-h7",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_tracked_paths(root: Path) -> set[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        Path(raw.decode()) for raw in completed.stdout.split(b"\0") if raw
    }


def parse_repo_path(value: object, label: str) -> tuple[Path | None, list[str]]:
    if not isinstance(value, str) or not value:
        return None, [f"{label} must be a non-empty repository-relative path"]
    if "\\" in value or value.startswith("/"):
        return None, [f"{label} must use a relative POSIX path"]
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return None, [f"{label} contains an unsafe path component"]
    pure = PurePosixPath(value)
    return Path(*pure.parts), []


def validate_sha256(value: object, label: str) -> list[str]:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        return [f"{label} must be a lowercase 64-character SHA-256"]
    return []


def validate_hashed_path(
        record: object,
        label: str,
        root: Path,
) -> tuple[Path | None, list[str]]:
    if not isinstance(record, dict):
        return None, [f"{label} must be an object"]
    expected_fields = {"path", "sha256"}
    errors: list[str] = []
    if set(record) != expected_fields:
        errors.append(
            f"{label} fields={sorted(record)}, expected {sorted(expected_fields)}")
    relative, path_errors = parse_repo_path(record.get("path"), f"{label}.path")
    errors.extend(path_errors)
    errors.extend(validate_sha256(record.get("sha256"), f"{label}.sha256"))
    if relative is None:
        return None, errors

    path = root / relative
    if not path.is_file():
        errors.append(f"{label} path does not exist: {relative}")
    elif not errors:
        actual = sha256_file(path)
        if actual != record["sha256"]:
            errors.append(
                f"{label} SHA-256 mismatch for {relative}: "
                f"manifest={record['sha256']}, actual={actual}")
    return relative, errors


def validate_sources(
        artifact: dict[str, object],
        artifact_path: Path,
        collector_path: Path | None,
        root: Path,
        tracked_paths: set[Path],
) -> list[str]:
    errors: list[str] = []
    capture_root, root_errors = parse_repo_path(
        artifact.get("source_capture_root"), "source_capture_root")
    errors.extend(root_errors)
    if artifact.get("source_tracking") != "gitignored_not_shipped":
        errors.append("source_tracking must be 'gitignored_not_shipped'")
    if artifact.get("source_relation") != "byte_identical_reconstruction_verified":
        errors.append(
            "source_relation must be 'byte_identical_reconstruction_verified'")

    captures = artifact.get("source_captures")
    if not isinstance(captures, list) or not captures:
        return errors + ["source_captures must be a non-empty list"]

    listed: set[Path] = set()
    capture_fields = {"path", "sha256"}
    for index, capture in enumerate(captures):
        label = f"source_captures[{index}]"
        if not isinstance(capture, dict):
            errors.append(f"{label} must be an object")
            continue
        if set(capture) != capture_fields:
            errors.append(
                f"{label} fields={sorted(capture)}, "
                f"expected {sorted(capture_fields)}")
        relative, path_errors = parse_repo_path(
            capture.get("path"), f"{label}.path")
        errors.extend(path_errors)
        hash_errors = validate_sha256(capture.get("sha256"), f"{label}.sha256")
        errors.extend(hash_errors)
        if relative is None:
            continue
        if relative in listed:
            errors.append(f"{label} duplicates source path {relative}")
        listed.add(relative)
        if capture_root is not None:
            root_parts = capture_root.parts
            if relative.parts[:len(root_parts)] != root_parts:
                errors.append(f"{label} is outside source_capture_root")
        if relative in tracked_paths:
            errors.append(f"{label} is marked gitignored but is tracked: {relative}")
        source_path = root / relative
        if source_path.exists() and not source_path.is_file():
            errors.append(f"{label} is not a regular file: {relative}")
        elif source_path.is_file() and not hash_errors:
            actual = sha256_file(source_path)
            if actual != capture["sha256"]:
                errors.append(
                    f"{label} SHA-256 mismatch for {relative}: "
                    f"manifest={capture['sha256']}, actual={actual}")

    summary_path = root / artifact_path
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, UnicodeError, ValueError) as exc:
        return errors + [f"cannot read source log names from {artifact_path}: {exc}"]
    configs = summary.get("configs") if isinstance(summary, dict) else None
    if not isinstance(configs, list) or capture_root is None:
        return errors + [f"{artifact_path} does not contain a configs list"]

    expected: set[Path] = set()
    for index, config in enumerate(configs):
        log_file = config.get("log_file") if isinstance(config, dict) else None
        if (not isinstance(log_file, str) or not log_file
                or PurePosixPath(log_file).name != log_file):
            errors.append(f"configs[{index}].log_file is not a safe basename")
            continue
        source_path = capture_root / log_file
        if source_path in expected:
            errors.append(f"configs[{index}].log_file duplicates {log_file}")
        expected.add(source_path)
    for missing in sorted(expected - listed):
        errors.append(f"missing source capture {missing}")
    for extra in sorted(listed - expected):
        errors.append(f"unexpected source capture {extra}")

    if (collector_path is not None
            and listed == expected
            and expected
            and all((root / source).is_file() for source in expected)):
        with tempfile.TemporaryDirectory() as directory:
            rebuilt = Path(directory) / "summary.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / collector_path),
                    str(root / capture_root),
                    "-o",
                    str(rebuilt),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                errors.append(
                    "source reconstruction failed: "
                    f"{completed.stdout}{completed.stderr}".strip())
            elif rebuilt.read_bytes() != summary_path.read_bytes():
                errors.append(
                    "source reconstruction is not byte-identical to "
                    f"{artifact_path}")
    return errors


def all_source_captures_present(document: object, root: Path) -> bool:
    if not isinstance(document, dict):
        return False
    artifacts = document.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    found = False
    for artifact in artifacts:
        captures = artifact.get("source_captures") if isinstance(artifact, dict) else None
        if not isinstance(captures, list) or not captures:
            return False
        for index, capture in enumerate(captures):
            value = capture.get("path") if isinstance(capture, dict) else None
            relative, errors = parse_repo_path(
                value, f"source_captures[{index}].path")
            if errors or relative is None or not (root / relative).is_file():
                return False
            found = True
    return found


def validate_repository_revision(
        root: Path,
        revision: str,
        records: dict[Path, str],
) -> list[str]:
    errors: list[str] = []
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if available.returncode != 0:
        return [
            f"artifact_repository.revision {revision} is unavailable; "
            "fetch sufficient Git history to verify provenance"
        ]

    for relative, expected_hash in sorted(records.items()):
        completed = subprocess.run(
            ["git", "show", f"{revision}:{relative.as_posix()}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            errors.append(
                f"artifact_repository.revision does not contain {relative}")
            continue
        actual_hash = sha256_bytes(completed.stdout)
        if actual_hash != expected_hash:
            errors.append(
                f"artifact_repository.revision SHA-256 mismatch for {relative}: "
                f"manifest={expected_hash}, revision={actual_hash}")
    return errors


def validate_execution_provenance(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["execution_provenance must be an object"]
    expected_fields = UNKNOWN_EXECUTION_FIELDS | {"clock_state"}
    errors: list[str] = []
    if set(value) != expected_fields:
        errors.append(
            "execution_provenance fields="
            f"{sorted(value)}, expected {sorted(expected_fields)}")
    for field in sorted(UNKNOWN_EXECUTION_FIELDS):
        if value.get(field) != "unknown":
            errors.append(
                f"execution_provenance.{field} must remain explicit 'unknown' "
                "until evidence is recorded")
    if value.get("clock_state") != "recorded_per_cell":
        errors.append(
            "execution_provenance.clock_state must be 'recorded_per_cell'")
    return errors


def validate_dependencies(
        value: object,
        root: Path,
) -> list[str]:
    if not isinstance(value, list):
        return ["declared_dependencies must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    fields = {
        "name", "revision", "basis", "evidence_path", "evidence_sha256",
    }
    for index, dependency in enumerate(value):
        label = f"declared_dependencies[{index}]"
        if not isinstance(dependency, dict):
            errors.append(f"{label} must be an object")
            continue
        if set(dependency) != fields:
            errors.append(
                f"{label} fields={sorted(dependency)}, expected {sorted(fields)}")
        name = dependency.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{label}.name must be a non-empty string")
        elif name in seen:
            errors.append(f"{label} duplicates dependency {name}")
        else:
            seen.add(name)

        evidence_record = {
            "path": dependency.get("evidence_path"),
            "sha256": dependency.get("evidence_sha256"),
        }
        evidence_path, evidence_errors = validate_hashed_path(
            evidence_record, f"{label}.evidence", root)
        errors.extend(evidence_errors)

        revision = dependency.get("revision")
        basis = dependency.get("basis")
        if basis == "declared_pin":
            if not isinstance(revision, str) or not revision or revision == "unknown":
                errors.append(f"{label}.revision must contain the declared pin")
            elif evidence_path is not None and (root / evidence_path).is_file():
                if revision not in (root / evidence_path).read_text():
                    errors.append(
                        f"{label}.revision is absent from its evidence file")
        elif basis == "unknown_mutable_default":
            if revision != "unknown":
                errors.append(
                    f"{label}.revision must be explicit 'unknown' for a mutable default")
        else:
            errors.append(f"{label}.basis has unsupported value {basis!r}")

    for missing in sorted(EXPECTED_DEPENDENCIES - seen):
        errors.append(f"missing declared dependency {missing}")
    for extra in sorted(seen - EXPECTED_DEPENDENCIES):
        errors.append(f"unexpected declared dependency {extra}")
    return errors


def validate_provenance(
        document: object,
        root: Path,
        tracked_paths: set[Path],
) -> list[str]:
    if not isinstance(document, dict):
        return ["top level must be an object"]
    top_fields = {
        "schema_version",
        "suite",
        "result_artifacts",
        "artifact_repository",
        "execution_provenance",
        "declared_dependencies",
    }
    errors: list[str] = []
    if set(document) != top_fields:
        errors.append(
            f"top-level fields={sorted(document)}, expected {sorted(top_fields)}")
    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if document.get("suite") != "cortex-m-deployability":
        errors.append("suite must be 'cortex-m-deployability'")

    expected_artifacts = {
        path for path in tracked_paths
        if path.name == RESULT_ARTIFACT_NAME and path.parent.name == "results"
    }
    artifacts = document.get("result_artifacts")
    listed_artifacts: set[Path] = set()
    revision_records: dict[Path, str] = {}
    artifact_fields = {
        "path",
        "kind",
        "sha256",
        "collector",
        "validator",
        "source_capture_root",
        "source_tracking",
        "source_relation",
        "source_captures",
    }
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("result_artifacts must be a non-empty list")
    else:
        for index, artifact in enumerate(artifacts):
            label = f"result_artifacts[{index}]"
            if not isinstance(artifact, dict):
                errors.append(f"{label} must be an object")
                continue
            if set(artifact) != artifact_fields:
                errors.append(
                    f"{label} fields={sorted(artifact)}, "
                    f"expected {sorted(artifact_fields)}")
            relative, path_errors = parse_repo_path(
                artifact.get("path"), f"{label}.path")
            errors.extend(path_errors)
            errors.extend(validate_sha256(
                artifact.get("sha256"), f"{label}.sha256"))
            if artifact.get("kind") != "device_result_summary":
                errors.append(f"{label}.kind must be 'device_result_summary'")
            if relative is None:
                continue
            if relative in listed_artifacts:
                errors.append(f"{label} duplicates result artifact {relative}")
            listed_artifacts.add(relative)
            artifact_path = root / relative
            if not artifact_path.is_file():
                errors.append(f"{label} path does not exist: {relative}")
            elif SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
                actual = sha256_file(artifact_path)
                if actual != artifact["sha256"]:
                    errors.append(
                        f"{label} SHA-256 mismatch for {relative}: "
                        f"manifest={artifact['sha256']}, actual={actual}")
                revision_records[relative] = artifact["sha256"]

            collector_path, collector_errors = validate_hashed_path(
                artifact.get("collector"), f"{label}.collector", root)
            validator_path, validator_errors = validate_hashed_path(
                artifact.get("validator"), f"{label}.validator", root)
            errors.extend(collector_errors)
            errors.extend(validator_errors)
            if collector_path is not None and not collector_errors:
                revision_records[collector_path] = artifact["collector"]["sha256"]
            if validator_path is not None and not validator_errors:
                revision_records[validator_path] = artifact["validator"]["sha256"]
            errors.extend(validate_sources(
                artifact, relative, collector_path, root, tracked_paths))

    for missing in sorted(expected_artifacts - listed_artifacts):
        errors.append(f"missing result artifact provenance {missing}")
    for extra in sorted(listed_artifacts - expected_artifacts):
        errors.append(f"unexpected result artifact provenance {extra}")

    repository = document.get("artifact_repository")
    repository_fields = {"revision", "basis"}
    if not isinstance(repository, dict):
        errors.append("artifact_repository must be an object")
    else:
        if set(repository) != repository_fields:
            errors.append(
                f"artifact_repository fields={sorted(repository)}, "
                f"expected {sorted(repository_fields)}")
        revision = str(repository.get("revision", ""))
        if not GIT_REVISION_RE.fullmatch(revision):
            errors.append("artifact_repository.revision must be a full Git SHA")
        if (repository.get("basis") !=
                "commit_containing_byte_identical_artifact_and_tools"):
            errors.append("artifact_repository.basis has an unsupported value")
        if GIT_REVISION_RE.fullmatch(revision):
            errors.extend(validate_repository_revision(
                root, revision, revision_records))

    errors.extend(validate_execution_provenance(
        document.get("execution_provenance")))
    errors.extend(validate_dependencies(
        document.get("declared_dependencies"), root))
    return errors
