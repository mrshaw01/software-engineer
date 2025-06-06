# Virtual Environments in Python

A **virtual environment** is a self-contained directory that contains a Python installation for a particular version of Python, plus a number of additional packages. Virtual environments help manage dependencies on a per-project basis and prevent conflicts between packages across projects.

## 1. Why Use a Virtual Environment?

Python projects often rely on third-party packages. When multiple projects require different versions of the same library, installing everything globally can result in:

- **Conflicting dependencies**
- **Broken environments** after upgrading packages
- **Unintended side effects** across unrelated projects

Virtual environments solve this by:

- Creating **isolated Python environments** per project
- Allowing independent package installations
- Avoiding dependency hell

## 2. Basic Tool: `virtualenv`

`virtualenv` is a third-party tool to create isolated Python environments.

### Installation

```bash
pip install virtualenv
```

### Creating a Virtual Environment

```bash
virtualenv myproject
```

- This creates a `myproject/` folder containing its own Python binaries and `site-packages`.

### Activating the Environment

```bash
source myproject/bin/activate  # On macOS/Linux
myproject\Scripts\activate     # On Windows
```

Your shell prompt will change to indicate the active environment.

### Deactivating

```bash
deactivate
```

Returns you to the global Python environment.

## 3. Using System Site Packages (Optional)

If you want your virtual environment to access globally installed packages:

```bash
virtualenv --system-site-packages myproject
```

This is rarely needed in modern projects, but useful for shared libraries or reducing duplication.

## 4. Modern Alternative: `venv` (Python 3.3+)

Python 3.3+ includes the `venv` module in the standard library. It’s preferred for most modern use cases.

### Creating and Activating

```bash
python3 -m venv myproject
source myproject/bin/activate
```

### Why Use `venv` Instead of `virtualenv`?

| Feature           | `venv` (standard) | `virtualenv` (3rd party) |
| ----------------- | ----------------- | ------------------------ |
| Built-in          | ✅ Yes            | ❌ Needs install         |
| Python 2.x        | ❌ No             | ✅ Yes                   |
| Lightweight       | ✅ Yes            | ✅ Yes                   |
| Advanced features | ❌ Basic          | ✅ More CLI options      |

If you're only using Python 3, prefer `venv`.

## 5. Virtual Environment Folder Structure

A virtual environment typically includes:

```
myproject/
├── bin/             # Executables (python, pip)
├── include/         # C headers
├── lib/             # Installed packages
└── pyvenv.cfg       # Config file
```

## 6. Best Practices

- Use one virtual environment per project.
- Store your virtual environment outside the project repo (or add to `.gitignore`).
- Use `requirements.txt` or `pyproject.toml` to record dependencies.

### Example:

```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

## 7. Smarter Environment Management

### Smart Directory Activation (Optional)

You can use tools like [`smartcd`](https://github.com/cxreg/smartcd) or shell hooks in `.bashrc`/`.zshrc` to automatically activate environments when changing into a project directory.

Alternatively, use [direnv](https://direnv.net/) for secure directory-based environment management.

## 8. Advanced Alternatives

| Tool       | Description                                                          |
| ---------- | -------------------------------------------------------------------- |
| **pipenv** | Combines `pip` and `virtualenv`, manages `Pipfile` and lock files    |
| **poetry** | Dependency + packaging tool for modern Python projects               |
| **conda**  | Popular in data science and ML; manages Python + binary dependencies |

### Poetry Example

```bash
poetry new myproject
cd myproject
poetry install
poetry shell
```

## 9. Platform Notes

- On **Windows**, activate via `.\myenv\Scripts\activate.bat`
- On **Linux/macOS**, use `source myenv/bin/activate`
- Use Python version managers like `pyenv` or `asdf` to manage multiple Python versions and isolate per-project Python versions.

## 10. Summary

| Task                     | Command                           |
| ------------------------ | --------------------------------- |
| Create venv (Python 3)   | `python3 -m venv env`             |
| Create venv (virtualenv) | `virtualenv env`                  |
| Activate (Linux/macOS)   | `source env/bin/activate`         |
| Activate (Windows)       | `.\env\Scripts\activate.bat`      |
| Deactivate               | `deactivate`                      |
| Save dependencies        | `pip freeze > requirements.txt`   |
| Restore dependencies     | `pip install -r requirements.txt` |

## 11. Resources

- [`venv` documentation (Python official)](https://docs.python.org/3/library/venv.html)
- [`virtualenv` on PyPI](https://pypi.org/project/virtualenv/)
- [`poetry` official docs](https://python-poetry.org/)
- [`pipenv` documentation](https://pipenv.pypa.io/en/latest/)
