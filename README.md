# RAGKit

A modular framework for building Retrieval-Augmented Generation (RAG)
applications using local Large Language Models.

## Goals

- Learn LLM engineering from first principles
- Build a production-quality RAG framework
- Support multiple document sources
- Support multiple vector databases
- Support multiple embedding models
- Support multiple LLM providers

# RAGKit - Python Environment Setup

This guide explains how to set up a Python 3.12 development environment using **pyenv** and a **virtual environment** on macOS.

## Prerequisites

- macOS
- Homebrew installed

---

## 1. Update Homebrew

```bash
brew update
```

---

## 2. Install pyenv

```bash
brew install pyenv
```

Verify the installation:

```bash
pyenv --version
```

---

## 3. Configure pyenv

Open your shell configuration file:

```bash
nano ~/.zshrc
```

Add the following lines to the end of the file:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Save and close the file.

Reload your shell:

```bash
source ~/.zshrc
```

---

## 4. Verify pyenv Installation

List installed Python versions:

```bash
pyenv versions
```

View available Python 3.12 releases:

```bash
pyenv install --list | grep "3.12"
```

Install Python 3.12.11:

```bash
pyenv install 3.12.11
```

Verify the installation:

```bash
pyenv versions
```

Expected output:

```text
* system
  3.12.11
```

---

## 5. Create the Project

Create a new project directory:

```bash
mkdir RAGKit
cd RAGKit
```

Set Python 3.12.11 for this project:

```bash
pyenv local 3.12.11
```

This creates a `.python-version` file in the project root.

---

## 6. Create a Virtual Environment

Create the virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Verify the active Python executable:

```bash
which python
```

Expected output:

```text
<project-directory>/.venv/bin/python
```

Verify the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.12.11
```

---

## Project Structure

```text
RAGKit/
├── .python-version
├── .venv/
└── README.md
```

---

## Daily Usage

### Activate the virtual environment

```bash
source .venv/bin/activate
```

### Verify Python version

```bash
python --version
```

### Verify Python executable

```bash
which python
```

### Deactivate the virtual environment

```bash
deactivate
```

---

## Useful pyenv Commands

### List installed Python versions

```bash
pyenv versions
```

### List available Python versions

```bash
pyenv install --list
```

### Install a Python version

```bash
pyenv install <version>
```

Example:

```bash
pyenv install 3.12.11
```

### Set Python version for the current project

```bash
pyenv local 3.12.11
```

### Set the global Python version

```bash
pyenv global 3.12.11
```

### Check the current Python version

```bash
python --version
```

---

## Next Steps

After the environment is ready, you can install project dependencies:

```bash
pip install -r requirements.txt
```

or, if using Ruff:

```bash
pip install ruff
ruff check .
```
