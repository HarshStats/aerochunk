<!-- Shields/Badges -->
![PyPI Version](https://img.shields.io/pypi/v/aerochunk.svg)
![Python Version](https://img.shields.io/pypi/pyversions/aerochunk.svg)
![License](https://img.shields.io/pypi/l/aerochunk.svg)

# AeroChunk: A Zero-Framework Semantic Chunker for RAG

## 1. Executive Summary

AeroChunk is a low-latency, zero-framework semantic text chunker engineered for Retrieval-Augmented Generation (RAG) pipelines. The efficacy of RAG systems is critically dependent on the quality of text segmentation. Prevailing methodologies often employ naive, fixed-size chunking (e.g., recursive character splitting), which can lead to significant semantic loss by arbitrarily severing conceptual units. Conversely, solutions that do preserve semantic integrity are frequently encumbered by heavy frameworks (e.g., LangChain, LlamaIndex) or reliant on costly, high-latency cloud APIs.

AeroChunk addresses this dichotomy by providing a high-fidelity semantic chunking mechanism that operates entirely on local machine resources. It leverages the computational efficiency of `sentence-transformers` and `numpy` to eliminate external dependencies, API costs, and framework bloat. A key innovation is the **Visual HTML Debugger**, an industry-first tool that generates an HTML report to provide a transparent, interpretable view of the semantic boundaries identified during the chunking process.

## 2. Empirical Benchmarks

AeroChunk was benchmarked against two standard LangChain text splitters on a repetitive text block to evaluate performance and chunking quality. The `all-MiniLM-L6-v2` model was used for all semantic comparisons to ensure a fair evaluation.

| Method                  | Execution Time (s) | Chunks Generated | Outcome         |
| ----------------------- | ------------------ | ---------------- | --------------- |
| **AeroChunk**           | **7.51**           | **72**           | **Optimal**     |
| LangChain (Recursive)   | 0.00               | 51               | Semantic Loss   |
| LangChain (Semantic)    | 7.53               | 1                | Failed to Split |

**Analysis:**
-   **AeroChunk** produced semantically coherent chunks at a competitive execution time.
-   **LangChain's RecursiveCharacterTextSplitter** was fast but failed to preserve semantic boundaries, resulting in fragmented and contextually poor chunks.
-   **LangChain's SemanticChunker** failed to identify any valid split points in the document, returning the entire text as a single chunk.

## 3. Architectural Methodology

The AeroChunk pipeline is a four-stage process designed for computational efficiency and semantic accuracy.

1.  **Regex Tokenization:** The input text is first segmented into individual sentences using a regular expression that identifies sentence-terminating punctuation (`.`, `!`, `?`).
2.  **Vectorization:** Each sentence is then converted into a 384-dimensional vector embedding using a local `sentence-transformers` model (`all-MiniLM-L6-v2` by default).
3.  **Cosine Similarity Analysis:** The system computes the pairwise cosine similarity between the vector embeddings of adjacent sentences. This score quantifies the semantic relatedness between them.
4.  **Threshold Bounding:** A semantic boundary is declared wherever the cosine similarity between two consecutive sentences drops below a predefined threshold (default: `0.5`). Sentences are aggregated into a chunk until such a drop is detected, at which point a new chunk begins.

This methodology ensures that chunks are formed from contiguous, semantically related sentences, thereby preserving the conceptual integrity of the source text.

## 4. Installation

Install the package using pip:
```bash
pip install aerochunk
```

## 5. Usage

### Basic Chunking

Instantiate the `AeroChunker` and pass text to the `chunk_text` method.

```python
from aerochunk import AeroChunker

# Your text document
text = """Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms. These algorithms can learn from data and generalize to unseen data. Recently, artificial neural networks have been able to surpass many previous approaches in performance."""

# Initialize the chunker
chunker = AeroChunker()

# Process the text
chunks = chunker.chunk_text(text)

# Print the resulting chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")
```

### Visual Debugging

After chunking, call the `export_debug_html()` method to generate a visual report of the chunking decisions. This is useful for fine-tuning the similarity threshold.

```python
# (Continuing from the previous example)

# Generate the HTML report
# This file will show where semantic boundaries were drawn and why.
debug_file = chunker.export_debug_html(output_file="aero_debug_report.html")

print(f"Visual debug report saved to: {debug_file}")
```

## 6. License

This project is licensed under the **MIT License**.