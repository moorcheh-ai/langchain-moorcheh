import json
import sys
from typing import Dict

# Updated to match the actual project structure
LIB_DIRS = ["libs/moorcheh"]

if __name__ == "__main__":
    files = sys.argv[1:]

    dirs_to_run: Dict[str, set] = {
        "lint": set(),
        "test": set(),
    }

    if len(files) == 300:
        # max diff length is 300 files - there are likely files missing
        raise ValueError("Max diff reached. Please manually run CI on changed libs.")

    for file in files:
        if any(
            file.startswith(dir_)
            for dir_ in (
                ".github/workflows",
                ".github/tools",
                ".github/actions",
                ".github/scripts/check_diff.py",
            )
        ):
            # add all LANGCHAIN_DIRS for infra changes
            dirs_to_run["test"].update(LIB_DIRS)

        # Check if file is in the main package directory or tests
        if any(file.startswith(dir_) for dir_ in ["libs/moorcheh/langchain_moorcheh/", "libs/moorcheh/tests/", "libs/moorcheh/docs/"]):
            dirs_to_run["test"].add("libs/moorcheh")
        elif file.startswith("libs/"):
            # This is now expected for our structure
            dirs_to_run["test"].add("libs/moorcheh")

    outputs = {
        "dirs-to-lint": list(dirs_to_run["lint"] | dirs_to_run["test"]),
        "dirs-to-test": list(dirs_to_run["test"]),
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201
