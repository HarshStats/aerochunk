import pytest
from aerochunk import AeroChunker
import os

@pytest.fixture(scope="module")
def chunker():
    # Initializes once for all tests to save time
    return AeroChunker()

def test_basic_chunking(chunker):
    text = "This is a sentence. This is another sentence! Totally unrelated topic here."
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0
    assert isinstance(chunks, list)

def test_empty_string(chunker):
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("    ") == []

def test_invalid_input(chunker):
    with pytest.raises(TypeError):
        chunker.chunk_text(12345)
    
    with pytest.raises(TypeError):
        chunker.chunk_text(None)

def test_export_without_chunking(chunker):
    fresh_chunker = AeroChunker()
    with pytest.raises(ValueError):
        fresh_chunker.export_debug_html()