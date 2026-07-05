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
---
---

# RAGKit Architecture - Over View

## Core Models

* Document
* Chunk
* Embedding
* QueryEmbedding
* SearchResult
* LLMResponse
* IndexingResult
* SourceDocument

---

## Interfaces

* Chunker
* Embedder
* Exception
* Indexer
* LLM
* Loader
* Pipeline
* PromptBuilder
* Retriever
* VectorStore

---

## Concrete Implementations

* TextLoader
* CharacterChunker
* OllamaEmbedder
* SentenceTransformerEmbedder
* ChromaVectorStore
* SimilarityRetriever
* DefaultPromptBuilder
* OllamaLLM
* RetrievalPipeline
* and many more..

---
---

# Read or study project in following sequence it will help to understand.

models/source_document.py<br>
sources/source.py<br>
sources/local_source.py<br> 
<br>
models/document.py<br>
loaders/loader.py<br>
loaders/text_loader.py<br>
loaders/loader_factory.py<br>
<br>
models/chunk.py<br>
chunker/chunker.py<br>
chunker/character_chunker.py<br>
<br>
models/embedding.py<br>
embeddings/embedder.py<br>
embeddings/ollama_embedding.py<br>
<br>
models/search_result.py<br>
vectorstores/vector_store.py<br>
vectorstores/chroma_vector_store.py<br>
<br>
retrievers/retriever.py<br>
retrievers/similarity_retriever.py<br>
<br>
prompts/default_prompt_builder.py<br>
prompts/prompt_builder.py<br>
<br>
llms/llm.py<br>
llms/ollama_llm.py<br>
<br>
pipelines/pipeline.py<br>
pipelines/retrieval_pipeline.py<br>
<br>
indexers/document_indexer.py<br>
indexers/indexer.py<br>
models/indexing_result.py<br>

---
## Few Q & A

### 1] Why so many interfaces?
<B>Ans. :</B>  Because everything is replaceable. Instead of OllamaEmbedder you can plug in OpenAIEmbedder without changing anything else.
Instead of ChromaVectorStore plug FAISS Nothing changes. That's the Dependency Inversion Principle.

### 2] Why SearchResult?
<B>Ans. :</B> Instead of returning Chunk you return Chunk + score Tomorrow Chunk score distance rerank_score retriever_name No API changes.

### 3] Why LLMResponse?
<B>Ans. :</B> Instead of str you return LLMResponse Tomorrow content token_usage latency finish_reason provider Still no API changes.

### 4] Why PromptBuilder?
<B>Ans. :</B> Without PromptBuilder Pipeline would contain
```bash
context="\n".join(...)
prompt=f"..." 
```
That violates SRP. Now Pipeline -> PromptBuilder Cleaner.

### 5] Why Retriever?
<B>Ans. :</B> Without Retriever Pipeline must know Embedder + VectorStore Now Pipeline -> Retriever Pipeline knows less.

### 6] Why Pipeline?
<B>Ans. :</B> Without Pipeline Every user writes embed(), search(), prompt(), generate() With Pipeline pipeline.invoke(query) That's the public API.