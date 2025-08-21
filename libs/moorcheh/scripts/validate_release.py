#!/usr/bin/env python3
"""
Validation script to ensure all components are ready for release.
Run this before pushing to main branch to catch issues early.
"""

import os
import sys
import subprocess
import tomllib
from pathlib import Path

def check_python_syntax():
    """Check that all Python files have valid syntax."""
    print("🔍 Checking Python syntax...")
    
    # Look for Python files in the libs/moorcheh directory
    package_dir = Path("libs/moorcheh")
    if not package_dir.exists():
        print("❌ libs/moorcheh directory not found")
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
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file, 'exec')
        except SyntaxError as e:
            errors.append(f"{py_file}:{e.lineno}: {e}")
        except Exception as e:
            errors.append(f"{py_file}: {e}")
    
    if errors:
        print("❌ Syntax errors found:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def check_pyproject_toml():
    """Check pyproject.toml configuration."""
    print("🔍 Checking pyproject.toml...")
    
    try:
        with open("libs/moorcheh/pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        
        # Check required fields
        required_fields = ["name", "version", "description"]
        for field in required_fields:
            if field not in config.get("project", {}):
                print(f"❌ Missing required field: project.{field}")
                return False
        
        # Check build system
        if "build-system" not in config:
            print("❌ Missing build-system configuration")
            return False
        
        print("✅ pyproject.toml configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error reading pyproject.toml: {e}")
        return False

def check_github_workflows():
    """Check GitHub Actions workflow files."""
    print("🔍 Checking GitHub Actions workflows...")
    
    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists():
        print("❌ .github/workflows directory not found")
        return False
    
    required_workflows = [
        "check_diffs.yml",
        "_lint.yml", 
        "_test.yml",
        "_release.yml",
        "_test_release.yml"
    ]
    
    missing = []
    for workflow in required_workflows:
        if not (workflow_dir / workflow).exists():
            missing.append(workflow)
    
    if missing:
        print(f"❌ Missing required workflows: {missing}")
        return False
    
    print("✅ All required workflow files present")
    return True

def check_project_structure():
    """Check that project structure matches workflow expectations."""
    print("🔍 Checking project structure...")
    
    # Check that libs/moorcheh directory exists
    if not Path("libs/moorcheh").exists():
        print("❌ libs/moorcheh directory not found")
        return False
    
    # Check that package directory exists
    if not Path("libs/moorcheh/langchain_moorcheh").exists():
        print("❌ langchain_moorcheh package directory not found")
        return False
    
    # Check that tests directory exists
    if not Path("libs/moorcheh/tests").exists():
        print("❌ tests directory not found")
        return False
    
    # Check that pyproject.toml exists in libs/moorcheh
    if not Path("libs/moorcheh/pyproject.toml").exists():
        print("❌ pyproject.toml not found in libs/moorcheh")
        return False
    
    print("✅ Project structure is correct")
    return True

def check_poetry_config():
    """Check Poetry configuration."""
    print("🔍 Checking Poetry configuration...")
    
    try:
        result = subprocess.run(
            ["poetry", "--version"], 
            capture_output=True, 
            text=True, 
            check=True,
            cwd="libs/moorcheh"
        )
        print(f"✅ Poetry version: {result.stdout.strip()}")
        
        # Check if poetry.lock exists
        if Path("libs/moorcheh/poetry.lock").exists():
            print("✅ poetry.lock file exists")
        else:
            print("⚠️  poetry.lock file not found (run 'poetry install' first)")
        
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Poetry not installed or not in PATH")
        return False
    except FileNotFoundError:
        print("❌ Poetry command not found")
        return False

def check_imports():
    """Check that the package can be imported (if dependencies are available)."""
    print("🔍 Checking package imports...")
    
    try:
        # Try to import the package from libs/moorcheh
        sys.path.insert(0, "libs/moorcheh")
        import langchain_moorcheh
        print("✅ Package imports successfully")
        
        # Check if main components can be imported
        try:
            from langchain_moorcheh import MoorchehVectorStore
            print("✅ MoorchehVectorStore imports successfully")
        except ImportError as e:
            print(f"⚠️  MoorchehVectorStore import failed: {e}")
            
        return True
    except ImportError as e:
        print(f"⚠️  Package import failed (likely missing dependencies): {e}")
        print("   This is expected if dependencies aren't installed yet")
        return True  # Not a critical failure
    except Exception as e:
        print(f"❌ Unexpected error importing package: {e}")
        return False

def check_gitignore():
    """Check that .gitignore files are properly configured."""
    print("🔍 Checking .gitignore configuration...")
    
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
        print("❌ .gitignore issues found:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("✅ .gitignore files properly configured")
        return True

def check_version_consistency():
    """Check that version is consistent across files."""
    print("🔍 Checking version consistency...")
    
    try:
        with open("libs/moorcheh/pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        
        project_version = config.get("project", {}).get("version")
        poetry_version = config.get("tool", {}).get("poetry", {}).get("version")
        
        if not project_version:
            print("❌ Missing project.version in pyproject.toml")
            return False
            
        if not poetry_version:
            print("❌ Missing tool.poetry.version in pyproject.toml")
            return False
            
        if project_version != poetry_version:
            print(f"❌ Version mismatch: project.version={project_version}, tool.poetry.version={poetry_version}")
            return False
            
        print(f"✅ Version consistent across files: {project_version}")
        return True
        
    except Exception as e:
        print(f"❌ Error checking version consistency: {e}")
        return False

def check_documentation():
    """Check that required documentation exists."""
    print("🔍 Checking documentation...")
    
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
        print("❌ Documentation issues found:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("✅ Required documentation present")
        return True

def check_clean_workspace():
    """Check that workspace is clean (no build artifacts)."""
    print("🔍 Checking workspace cleanliness...")
    
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
        print("⚠️  Workspace cleanliness issues:")
        for issue in issues:
            print(f"   {issue}")
        print("   Consider running: git clean -fdx")
        return True  # Not critical, just a warning
    else:
        print("✅ Workspace is clean")
        return True

def main():
    """Run all validation checks."""
    print("🚀 Running release validation checks...\n")
    
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
            print()
        except Exception as e:
            print(f"❌ Check failed with error: {e}\n")
    
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your project is ready for release.")
        print("\nNext steps:")
        print("1. Commit and push your changes")
        print("2. The CI/CD pipeline will automatically run")
        print("3. If all tests pass, you can trigger a release")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues before releasing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
