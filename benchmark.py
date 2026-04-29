import time
import matplotlib.pyplot as plt
from aerochunk import AeroChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Prepare Sample Data (Simulating a long document)
sample_text = """Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms. 
These algorithms can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Recently, artificial neural networks have been able to surpass many previous approaches in performance. 
The mathematical foundations of ML are provided by mathematical optimization (mathematical programming) methods. Data mining is a related (parallel) field of study, focusing on exploratory data analysis (EDA) through unsupervised learning. 
From a theoretical point of view, probably approximately correct (PAC) learning provides a framework for describing machine learning. History of machine learning is intertwined with the history of artificial intelligence. 
Modern machine learning has two objectives: classify data based on models which have been developed, and make predictions for future outcomes based on these models. A hypothetical algorithm specific to classifying data may use computer vision of moles coupled with supervised learning in order to train it to classify the cancerous moles.""" * 10

print("Starting Benchmark...")

# 2. Test AeroChunk (Your Library)
start_time = time.time()
aero = AeroChunker()
aero_chunks = aero.chunk_text(sample_text)
aero_time = time.time() - start_time

# Optional: Generate your unique HTML debug file
aero.export_debug_html("aero_debug.html")

# 3. Test LangChain (Recursive Character - The standard naive band-aid)
start_time = time.time()
lc_recursive = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
lc_recursive_chunks = lc_recursive.split_text(sample_text)
lc_recursive_time = time.time() - start_time

# 4. Test LangChain (Semantic Chunker - The heavy framework alternative)
start_time = time.time()
# Using the exact same local open-source model for a fair comparison
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
lc_semantic = SemanticChunker(hf_embeddings)
lc_semantic_chunks = lc_semantic.create_documents([sample_text])
lc_semantic_time = time.time() - start_time

# 5. Output Data Table to Terminal
print("\n" + "="*50)
print(f"{'Library':<25} | {'Time (Seconds)':<15} | {'Chunks Generated'}")
print("="*50)
print(f"{'AeroChunk (Yours)':<25} | {aero_time:<15.4f} | {len(aero_chunks)}")
print(f"{'LangChain (Recursive)':<25} | {lc_recursive_time:<15.4f} | {len(lc_recursive_chunks)}")
print(f"{'LangChain (Semantic)':<25} | {lc_semantic_time:<15.4f} | {len(lc_semantic_chunks)}")
print("="*50)

# 6. Generate and Save Plot
labels = ['AeroChunk\n(Yours)', 'LangChain\n(Recursive)', 'LangChain\n(Semantic)']
times = [aero_time, lc_recursive_time, lc_semantic_time]

plt.figure(figsize=(8, 5))
colors = ['#4CAF50', '#FF9800', '#F44336']
plt.bar(labels, times, color=colors)
plt.title('Execution Time Comparison (Lower is Better)')
plt.ylabel('Time (Seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot to a file
plt.savefig('benchmark_results.png')
print("\nPlot saved as 'benchmark_results.png'.")
print("Visual debug HTML saved as 'aero_debug.html'.")