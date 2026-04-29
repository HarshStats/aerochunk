from sentence_transformers import SentenceTransformer
import numpy as np
import re

class AeroChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.5):
        # Loads a tiny, fast local model. No API keys needed.
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def _split_sentences(self, text):
        return [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s]

    def chunk_text(self, text):
        sentences = self._split_sentences(text)
        if not sentences: return []

        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        similarity_drops = [] # Track this for the visual debugger

        for i in range(1, len(sentences)):
            # Calculate cosine similarity between consecutive sentences
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
        """The Unique Feature: Generates a visual report of the chunking decisions."""
        html = "<html><body style='font-family: Arial; line-height: 1.6;'>"
        html += "<h2>AeroChunk Visual Debugger</h2>"
        
        for idx, chunk in enumerate(self.last_chunks):
            html += f"<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>"
            html += f"<span style='color: #888; font-size: 0.8em;'>Chunk {idx + 1}</span><br>"
            html += f"{chunk}</div>"
            
            if idx < len(self.last_drops):
                html += f"<div style='text-align: center; color: red; font-size: 0.9em;'>"
                html += f"↓ Semantic Similarity Drop (Score: {self.last_drops[idx]:.2f}) ↓</div><br>"

        html += "</body></html>"
        
        with open(output_file, "w",encoding="utf-8") as f:
            f.write(html)
        return output_file