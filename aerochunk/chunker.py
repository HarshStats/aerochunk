from sentence_transformers import SentenceTransformer
import numpy as np
import re

class AeroChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.last_chunks = []
        self.last_drops = []

    def _split_sentences(self, text):
        return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s]

    def chunk_text(self, text, batch_size=32):
        # 1. Error Handling: Input Validation
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, but got {type(text).__name__}")
        
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences: 
            return []

        # 2. Memory Management: Batching
        # By explicitly setting batch_size, we prevent RAM overflow on 1000+ page PDFs
        embeddings = self.model.encode(sentences, batch_size=batch_size)
        
        chunks = []
        current_chunk = [sentences[0]]
        similarity_drops = [] 

        for i in range(1, len(sentences)):
            # Fast mathematical distance calculation
            sim = np.dot(embeddings[i-1], embeddings[i]) / (np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]))
            
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                similarity_drops.append(sim)
                current_chunk = [sentences[i]]
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        self.last_chunks = chunks
        self.last_drops = similarity_drops
        return chunks

    def export_debug_html(self, output_file="debug_aerochunk.html"):
        if not self.last_chunks:
            raise ValueError("No chunks found. Run chunk_text() before exporting debug data.")

        html = "<html><body style='font-family: Arial; line-height: 1.6;'>"
        html += "<h2>AeroChunk Visual Debugger</h2>"
        
        for idx, chunk in enumerate(self.last_chunks):
            html += f"<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>"
            html += f"<span style='color: #888; font-size: 0.8em;'>Chunk {idx + 1}</span><br>"
            html += f"{chunk}</div>"
            
            if idx < len(self.last_drops):
                html += f"<div style='text-align: center; color: red; font-size: 0.9em;'>"
                html += f"&#8595; Semantic Similarity Drop (Score: {self.last_drops[idx]:.2f}) &#8595;</div><br>"

        html += "</body></html>"
        
        # Using HTML entity &#8595; instead of the raw arrow character to avoid any Windows encoding errors
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        return output_file