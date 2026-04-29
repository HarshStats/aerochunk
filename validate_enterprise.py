from aerochunk import AeroChunker

# 1. Initialize with Enterprise defaults
chunker = AeroChunker(
    threshold=0.5, 
    min_sentences=2, 
    max_sentences=5, 
    overlap_sentences=1
)

# 2. The "Critic's Choice" Test String
# This contains acronyms, decimals, and pronoun dependencies
validation_text = (
    "Dr. Harsh Rana released AeroChunk v0.2.1 today. "
    "It increased the reliability of RAG pipelines by 45.5 percent. "
    "The library now uses spaCy for NLP. "
    "This shift ensures that abbreviations like U.S.A. are handled correctly. "
    "Furthermore, the overlap feature prevents context loss. "
    "It allows the LLM to remember the subject of the previous chunk. "
    "This makes the library production-ready."
)

print("--- Running Enterprise Validation ---")
result = chunker.chunk_text(validation_text)

# Check 1: Sentence Splitting Accuracy
print(f"\n1. Total Sentences detected: {len(chunker._split_sentences(validation_text))}")
# If this is 7, spaCy worked. If it's 10+, regex-style splitting failed.

# Check 2: Context Overlap
for i, chunk in enumerate(result.chunks):
    print(f"\n[Chunk {i+1}]:")
    print(f"Content: {chunk}")

# Check 3: Visual Debugger
result.export_debug_html("enterprise_validation.html")
print(f"\nValidation HTML generated: enterprise_validation.html")