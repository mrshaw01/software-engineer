# UV: A Modern Python Environment & Package Manager

[`uv`](https://docs.astral.sh/uv/) is an **extremely fast** tool for managing Python virtual environments, packages, and project dependencies. Built in Rust by the creators of `ruff`, `uv` aims to replace tools like `pip`, `venv`, `poetry`, and `pyenv`—all in one.

## 1. Why Use `uv`?

Traditional tools like `pip`, `venv`, and `poetry` work, but they can be:

- **Slow** during installation
- **Verbose** in CLI workflows
- **Inefficient** with dependency resolution

`uv` addresses these pain points:

- **10–100× faster** than pip/poetry
- Parallel package downloads
- Uses modern standards (`pyproject.toml`, `uv.lock`)
- Built in **Rust** for maximum performance

## 2. Installing `uv`

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Or manually download binaries from the [release page](https://github.com/astral-sh/uv/releases).

## 3. Quick Start

### Initialize a New Project

```bash
uv init --bare
```

This creates a `pyproject.toml` file without any source scaffolding.

### Add/Remove Dependencies

```bash
uv add requests
uv remove numpy
```

Lock file updates automatically (`uv.lock`).

### Sync/Install Project Dependencies

```bash
uv sync  # Like poetry install or pip install -r requirements.txt
```

### Run a Command in the Virtual Environment

```bash
uv run python script.py
```

You can also use:

```bash
uv run pytest
```

### Lock Dependencies

```bash
uv lock --upgrade-package pandas==2.2.2
```

## 4. Project Structure

Typical layout created by `uv init`:

```
myproject/
├── pyproject.toml     # Project metadata & dependencies
├── uv.lock            # Locked, reproducible dependencies
└── venv/              # Automatically managed virtualenv
```

> No need to manually run `python -m venv` or install `virtualenv`.

## 5. Comparison with Other Tools

| Feature                     | `uv`         | `poetry`         | `pip` + `venv` |
| --------------------------- | ------------ | ---------------- | -------------- |
| Speed                       | 🚀 Blazing   | 🐢 Slow          | 🐢 Slow        |
| Built-in venv management    | ✅ Yes       | ✅ Yes           | ❌ No          |
| Lock file support           | ✅ `uv.lock` | ✅ `poetry.lock` | ❌ No          |
| Compatible with `pyproject` | ✅ Yes       | ✅ Yes           | ⚠️ Limited     |
| Python version management   | ✅ Optional  | ❌ External      | ❌ External    |
| Parallel downloads          | ✅ Yes       | ❌ No            | ❌ No          |

## 6. Advanced Usage

### Run from Scripts

```bash
uv run python my_script.py
```

### Use with `ruff` Formatter

Since `uv` is from the same team as `ruff`, they work well together:

```bash
uv add ruff
uv run ruff src/
```

## 7. Best Practices

- Prefer `uv` for **fast, isolated, reproducible** environments.
- Always commit `pyproject.toml` and `uv.lock` for project reproducibility.
- Use `uv run` instead of activating/deactivating manually.

## 8. Resources

- [Official Docs](https://docs.astral.sh/uv/)
- [GitHub Project](https://github.com/astral-sh/uv)
- [PEP 621 - Project metadata via `pyproject.toml`](https://peps.python.org/pep-0621/)
- [Ruff (Linter/Formatter)](https://docs.astral.sh/ruff/)
