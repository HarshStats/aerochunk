# AeroChunk
A zero-framework, low-latency semantic chunker for RAG pipelines with visual HTML debugging.

Stop installing massive frameworks just to split text. AeroChunk relies entirely on local execution (`sentence-transformers` and `numpy`) to find semantic boundaries, resulting in zero API costs and zero dependency bloat.

## Why AeroChunk? 
In out-of-the-box testing on standard repetitive text, LangChain's SemanticChunker fails to identify proper thresholds, resulting in massive, unusable chunks. AeroChunk properly segments the data while matching LangChain's execution speed.

| Library | Time (Seconds) | Chunks Generated |
|---------|----------------|------------------|
| **AeroChunk** | **7.51s** | **72** |
| LangChain (Semantic) | 7.53s | 1 (Failed to split) |

## Features
* **Zero Framework Bloat:** No LangChain, no LlamaIndex.
* **Visual Debugging:** Call `.export_debug_html()` to generate an HTML file showing exactly where and why semantic boundaries were drawn in your text.