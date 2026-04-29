import tracemalloc
import time
from aerochunk import AeroChunker
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Create a massive wall of text to force high RAM usage (approx 15,000 words)
sample_text = """Machine learning represents a massive shift in how we process data. 
Unlike traditional algorithms that require explicit instructions, neural networks 
learn patterns independently. This requires significant computational overhead. """ * 500

print("Starting Stress Test (Measuring Peak RAM & Speed)...\n")

# --- 1. Test AeroChunk ---
tracemalloc.start()
start_time = time.time()

aero = AeroChunker()
# Using your new batching feature to keep memory low
aero_chunks = aero.chunk_text(sample_text, batch_size=32) 

_, peak_mem_aero = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_aero = time.time() - start_time
peak_mb_aero = peak_mem_aero / (1024 * 1024)

# --- 2. Test LangChain Semantic Chunker ---
tracemalloc.start()
start_time = time.time()

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
lc_semantic = SemanticChunker(hf_embeddings)
lc_chunks = lc_semantic.create_documents([sample_text])

_, peak_mem_lc = tracemalloc.get_traced_memory()
tracemalloc.stop()
time_lc = time.time() - start_time
peak_mb_lc = peak_mem_lc / (1024 * 1024)

# --- 3. Output Results ---
print("="*65)
print(f"{'Library':<20} | {'Peak RAM (MB)':<15} | {'Time (s)':<10} | {'Chunks'}")
print("="*65)
print(f"{'AeroChunk (Yours)':<20} | {peak_mb_aero:<15.2f} | {time_aero:<10.2f} | {len(aero_chunks)}")
print(f"{'LangChain':<20} | {peak_mb_lc:<15.2f} | {time_lc:<10.2f} | {len(lc_chunks)}")
print("="*65)