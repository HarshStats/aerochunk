import pytest
from aerochunk import AeroChunker
from aerochunk.chunker import ChunkingResult

@pytest.fixture(scope="module")
def chunker():
    return AeroChunker()

def test_basic_chunking(chunker):
    text = "This is a sentence. This is another sentence! Totally unrelated topic here."
    result = chunker.chunk_text(text)
    # Check that it returns the right object type
    assert isinstance(result, ChunkingResult)
    # Check that we actually got chunks
    assert len(result.chunks) > 0

def test_empty_string(chunker):
    # Update: Check the .chunks attribute of the result object
    result = chunker.chunk_text("")
    assert result.chunks == []
    
    result_space = chunker.chunk_text("    ")
    assert result_space.chunks == []

def test_invalid_input(chunker):
    with pytest.raises(TypeError):
        chunker.chunk_text(12345)

def test_export_logic(chunker):
    # Test that the result object handles the export, not the chunker itself
    result = chunker.chunk_text("Some text to chunk.")
    # This should work now
    path = result.export_debug_html("test_output.html")
    assert "test_output.html" in path

def test_stateless_architecture(chunker):
    # Verify the chunker itself doesn't store the data (Thread-safety check)
    assert not hasattr(chunker, 'last_chunks')