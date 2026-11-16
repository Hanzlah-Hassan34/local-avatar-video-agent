import json
import os
import subprocess
import threading
from pathlib import Path
from string import Formatter
from typing import Any


class WorkerError(RuntimeError):
    pass


class WorkerRunner:
    """Runs configured local model workers without invoking a shell."""

    _gpu_lock = threading.Lock()

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def configured_stages(self) -> set[str]:
        return {
            stage
            for stage, config in self._read_config().items()
            if self._is_configured(config)
        }

    @staticmethod
    def _is_configured(config: Any) -> bool:
        if not isinstance(config, dict):
            return False
        command = config.get("command")
        timeout = config.get("timeout_seconds", 1800)
        return (
            isinstance(command, list)
            and bool(command)
            and all(isinstance(part, str) for part in command)
            and isinstance(timeout, int)
            and timeout >= 1
        )

    def run(self, stage: str, payload: dict[str, Any]) -> None:
        config = self._read_config().get(stage)
        if config is None:
            raise WorkerError(f"No local worker is configured for stage '{stage}'.")
        command = config.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) for part in command
        ):
            raise WorkerError(f"Worker '{stage}' must define a non-empty command array.")

        spec_path = Path(payload["work_dir"]) / f"{stage}-{payload.get('position', 'job')}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        values = {key: str(value) for key, value in payload.items()}
        values["spec"] = str(spec_path.resolve())
        try:
            expanded_command = [
                self._format_argument(argument, values) for argument in command
            ]
        except (KeyError, ValueError) as error:
            raise WorkerError(f"Invalid field in '{stage}' worker command arguments: {error}") from error

        timeout = config.get("timeout_seconds", 1800)
        if not isinstance(timeout, int) or timeout < 1:
            raise WorkerError(f"Worker '{stage}' timeout_seconds must be a positive integer.")

        spec_path = spec_path.resolve()
        log_path = Path(payload["work_dir"]) / f"{stage}-{payload.get('position', 'job')}.log"
        try:
            with self._gpu_lock, log_path.open("w", encoding="utf-8", errors="replace") as log:
                result = subprocess.run(
                    expanded_command,
                    cwd=config.get("working_directory") or None,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    check=False,
                    shell=False,
                    env={
                        **os.environ,
                        "AVATAR_JOB_SPEC": str(spec_path),
                        "AVATAR_STAGE": stage,
                    },
                )
        except FileNotFoundError as error:
            raise WorkerError(
                f"Worker '{stage}' could not start; check its executable and working directory."
            ) from error
        except subprocess.TimeoutExpired as error:
            tail = self._read_log_tail(log_path)
            raise WorkerError(
                f"Worker '{stage}' exceeded its {timeout}s timeout. "
                f"Log: {log_path}. Output: {tail or '(no output)'}"
            ) from error
        if result.returncode:
            tail = self._read_log_tail(log_path)
            raise WorkerError(
                f"Worker '{stage}' exited with code {result.returncode}. "
                f"Log: {log_path}. Output: {tail or '(no output)'}"
            )

    @staticmethod
    def _read_log_tail(path: Path) -> str:
        with path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - 1500))
            return log.read().decode("utf-8", errors="replace").strip()

    def _read_config(self) -> dict[str, dict[str, Any]]:
        if not self.config_path.is_file():
            return {}
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkerError(f"Could not read worker configuration: {error}") from error
        if not isinstance(config, dict):
            raise WorkerError("Worker configuration must be a JSON object.")
        return config

    @staticmethod
    def _format_argument(argument: str, values: dict[str, str]) -> str:
        fields = [field for _, field, _, _ in Formatter().parse(argument) if field]
        missing = set(fields) - values.keys()
        if missing:
            raise KeyError(f"unknown placeholder(s): {', '.join(sorted(missing))}")
        return argument.format_map(values)
