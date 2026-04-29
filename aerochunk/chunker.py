import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
from typing import List

# NEW: A dedicated data class to hold results per-request. 
# This makes the main chunker completely stateless and thread-safe.
@dataclass
class ChunkingResult:
    chunks: List[str]
    similarity_drops: List[float]

    def export_debug_html(self, output_file="debug_aerochunk.html"):
        if not self.chunks:
            raise ValueError("No chunks found to export.")

        html = "<html><body style='font-family: Arial; line-height: 1.6;'>"
        html += "<h2>AeroChunk Visual Debugger</h2>"
        
        for idx, chunk in enumerate(self.chunks):
            html += f"<div style='border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>"
            html += f"<span style='color: #888; font-size: 0.8em;'>Chunk {idx + 1}</span><br>"
            html += f"{chunk}</div>"
            
            if idx < len(self.similarity_drops):
                drop = self.similarity_drops[idx]
                reason = f"Semantic Drop (Score: {drop:.2f})" if drop != 0.0 else "Max Sentence Ceiling Hit"
                html += f"<div style='text-align: center; color: red; font-size: 0.9em;'>&#8595; {reason} &#8595;</div><br>"

        html += "</body></html>"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        return output_file


class AeroChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.5, 
                 min_sentences=2, max_sentences=12, overlap_sentences=1):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            raise OSError("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        self.nlp.add_pipe("sentencizer")

    def _split_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def chunk_text(self, text, batch_size=32) -> ChunkingResult:
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, but got {type(text).__name__}")
        if not text.strip(): return ChunkingResult([], [])

        sentences = self._split_sentences(text)
        if not sentences: return ChunkingResult([], [])

        # The batch size handles CPU math overload. For massive files, 
        # a true streaming architecture would be needed, but this is safe for 99% of RAG PDFs.
        embeddings = self.model.encode(sentences, batch_size=batch_size)
        
        chunks = []
        current_chunk_sents = [sentences[0]]
        current_chunk_embs = [embeddings[0]]
        similarity_drops = []

        for i in range(1, len(sentences)):
            chunk_mean_emb = np.mean(current_chunk_embs, axis=0)
            sim = np.dot(chunk_mean_emb, embeddings[i]) / (np.linalg.norm(chunk_mean_emb) * np.linalg.norm(embeddings[i]))
            
            if len(current_chunk_sents) < self.min_sentences:
                current_chunk_sents.append(sentences[i])
                current_chunk_embs.append(embeddings[i])
            elif len(current_chunk_sents) >= self.max_sentences or sim < self.threshold:
                chunks.append(" ".join(current_chunk_sents))
                similarity_drops.append(sim if sim < self.threshold else 0.0)
                
                overlap_idx = max(0, len(current_chunk_sents) - self.overlap_sentences)
                current_chunk_sents = current_chunk_sents[overlap_idx:] + [sentences[i]]
                current_chunk_embs = current_chunk_embs[overlap_idx:] + [embeddings[i]]
            else:
                current_chunk_sents.append(sentences[i])
                current_chunk_embs.append(embeddings[i])

        if current_chunk_sents:
            final_text = " ".join(current_chunk_sents)
            if not chunks or final_text != chunks[-1]:
                chunks.append(final_text)

        # NEW: Return the stateless result object instead of saving to 'self'
        return ChunkingResult(chunks, similarity_drops)