#!/usr/bin/env python3
"""
Validation script to ensure all components are ready for release.
Run this before pushing to main branch to catch issues early.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import tomllib


def check_python_syntax():
    """Check that all Python files have valid syntax."""

    # Look for Python files in the libs/moorcheh directory
    package_dir = Path("libs/moorcheh")
    if not package_dir.exists():
        return False

    python_files = []
    for root, dirs, files in os.walk(package_dir):
        if "venv" in root or ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    errors = []
    for py_file in python_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                compile(f.read(), py_file, "exec")
        except SyntaxError as e:
            errors.append(f"{py_file}:{e.lineno}: {e}")
        except Exception as e:
            errors.append(f"{py_file}: {e}")

    if errors:
        for error in errors:
            pass
        return False
    else:
        return True


def check_pyproject_toml():
    """Check pyproject.toml configuration."""

    try:
        with open("libs/moorcheh/pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        # Check required fields
        required_fields = ["name", "version", "description"]
        for field in required_fields:
            if field not in config.get("project", {}):
                return False

        # Check build system
        if "build-system" not in config:
            return False

        return True

    except Exception:
        return False


def check_github_workflows():
    """Check GitHub Actions workflow files."""

    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists():
        return False

    required_workflows = [
        "check_diffs.yml",
        "_lint.yml",
        "_test.yml",
        "_release.yml",
        "_test_release.yml",
    ]

    missing = []
    for workflow in required_workflows:
        if not (workflow_dir / workflow).exists():
            missing.append(workflow)

    if missing:
        return False

    return True


def check_project_structure():
    """Check that project structure matches workflow expectations."""

    # Check that libs/moorcheh directory exists
    if not Path("libs/moorcheh").exists():
        return False

    # Check that package directory exists
    if not Path("libs/moorcheh/langchain_moorcheh").exists():
        return False

    # Check that tests directory exists
    if not Path("libs/moorcheh/tests").exists():
        return False

    # Check that pyproject.toml exists in libs/moorcheh
    if not Path("libs/moorcheh/pyproject.toml").exists():
        return False

    return True


def check_poetry_config():
    """Check Poetry configuration."""

    try:
        subprocess.run(
            ["poetry", "--version"],
            capture_output=True,
            text=True,
            check=True,
            cwd="libs/moorcheh",
        )

        # Check if poetry.lock exists
        if Path("libs/moorcheh/poetry.lock").exists():
            pass
        else:
            pass

        return True

    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def check_imports():
    """Check that the package can be imported (if dependencies are available)."""

    try:
        # Try to import the package from libs/moorcheh
        sys.path.insert(0, "libs/moorcheh")

        # Check if package can be found and imported
        spec = importlib.util.find_spec("langchain_moorcheh")
        if spec is None:
            return False

        # Check if main components can be imported
        spec = importlib.util.find_spec("langchain_moorcheh.vectorstores")
        if spec is None:
            pass

        return True
    except ImportError:
        return True  # Not a critical failure
    except Exception:
        return False


def check_gitignore():
    """Check that .gitignore files are properly configured."""

    errors = []

    # Check root .gitignore
    root_gitignore = Path(".gitignore")
    if not root_gitignore.exists():
        errors.append("Root .gitignore file missing")
    else:
        content = root_gitignore.read_text()
        required_patterns = ["dist/", "__pycache__/", ".venv", ".mypy_cache"]
        for pattern in required_patterns:
            if pattern not in content:
                errors.append(f"Root .gitignore missing pattern: {pattern}")

    # Check package .gitignore
    pkg_gitignore = Path("libs/moorcheh/.gitignore")
    if not pkg_gitignore.exists():
        errors.append("Package .gitignore file missing")
    else:
        content = pkg_gitignore.read_text()
        required_patterns = ["dist/", "__pycache__/", ".venv"]
        for pattern in required_patterns:
            if pattern not in content:
                errors.append(f"Package .gitignore missing pattern: {pattern}")

    if errors:
        for error in errors:
            pass
        return False
    else:
        return True


def check_version_consistency():
    """Check that version is consistent across files."""

    try:
        with open("libs/moorcheh/pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        project_version = config.get("project", {}).get("version")
        poetry_version = config.get("tool", {}).get("poetry", {}).get("version")

        if not project_version:
            return False

        if not poetry_version:
            return False

        if project_version != poetry_version:
            return False

        return True

    except Exception:
        return False


def check_documentation():
    """Check that required documentation exists."""

    errors = []

    # Check README files
    if not Path("README.md").exists():
        errors.append("Root README.md missing")

    if not Path("libs/moorcheh/README.md").exists():
        errors.append("Package README.md missing")

    # Check LICENSE
    if not Path("libs/moorcheh/LICENSE").exists():
        errors.append("LICENSE file missing")

    if errors:
        for error in errors:
            pass
        return False
    else:
        return True


def check_clean_workspace():
    """Check that workspace is clean (no build artifacts)."""

    issues = []

    # Check for build artifacts that should be gitignored
    if Path("dist").exists():
        issues.append("Root dist/ directory should be gitignored")

    if Path("libs/moorcheh/dist").exists():
        issues.append("Package dist/ directory should be gitignored")

    # Check for Python cache directories
    for cache_dir in Path(".").rglob("__pycache__"):
        issues.append(f"Python cache directory found: {cache_dir}")

    if issues:
        for issue in issues:
            pass
        return True  # Not critical, just a warning
    else:
        return True


def main():
    """Run all validation checks."""

    checks = [
        check_python_syntax,
        check_pyproject_toml,
        check_github_workflows,
        check_project_structure,
        check_poetry_config,
        check_imports,
        check_gitignore,
        check_version_consistency,
        check_documentation,
        check_clean_workspace,
    ]

    passed = 0
    total = len(checks)

    for check in checks:
        try:
            if check():
                passed += 1
        except Exception:
            pass

    if passed == total:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
