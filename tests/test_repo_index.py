"""Tests for the repo_index module."""

from pathlib import Path

from rfsn_controller.repo_index import FileInfo, RepoIndex, SymbolInfo

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "tiny_repo"


class TestRepoIndex:
    """Tests for RepoIndex class."""
    
    def test_build_from_fixture(self) -> None:
        """Test building an index from the fixture repo."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        assert len(index.files) >= 2
        assert any(f.path == "main.py" for f in index.files)
        assert any(f.path == "utils.py" for f in index.files)
    
    def test_file_info_includes_language(self) -> None:
        """Test that files have language detected."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        python_files = [f for f in index.files if f.language == "python"]
        assert len(python_files) >= 2
    
    def test_import_graph_built(self) -> None:
        """Test that import graph is extracted from Python files."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        # utils.py imports from main
        assert "utils.py" in index.import_graph
        assert "main" in index.import_graph["utils.py"]
    
    def test_symbols_extracted(self) -> None:
        """Test that symbols are extracted from Python files."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        # Should find the functions and class from main.py
        function_names = {s.name for s in index.symbols if s.kind == "function"}
        class_names = {s.name for s in index.symbols if s.kind == "class"}
        
        assert "hello_world" in function_names
        assert "add_numbers" in function_names
        assert "Calculator" in class_names
    
    def test_search_symbols(self) -> None:
        """Test symbol search functionality."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        # Search for "calc" should find calculator-related symbols
        results = index.search_symbols("calc")
        assert len(results) > 0
        assert any("Calculator" in s.name or "calc" in s.name.lower() for s in results)
    
    def test_to_json_and_back(self) -> None:
        """Test serialization and deserialization."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        json_data = index.to_json()
        
        assert "files" in json_data
        assert "symbols" in json_data
        assert "import_graph" in json_data
        assert json_data["file_count"] == len(index.files)
    
    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test saving and loading index from file."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        save_path = tmp_path / "index.json"
        index.save(str(save_path))
        
        loaded = RepoIndex.load(str(save_path))
        
        assert len(loaded.files) == len(index.files)
        assert len(loaded.symbols) == len(index.symbols)
    
    def test_compact_json_for_llm(self) -> None:
        """Test compact JSON output for LLM context."""
        index = RepoIndex.build(str(FIXTURES_DIR))
        
        compact = index.to_compact_json(max_files=10, max_symbols=20)
        
        assert "file_count" in compact
        assert "languages" in compact
        assert "top_files" in compact
        assert "symbols" in compact
    
    def test_deterministic_ordering(self) -> None:
        """Test that index output is deterministic."""
        index1 = RepoIndex.build(str(FIXTURES_DIR))
        index2 = RepoIndex.build(str(FIXTURES_DIR))
        
        # Files should be in same order
        assert [f.path for f in index1.files] == [f.path for f in index2.files]
        
        # Symbols should be in same order
        assert [(s.file, s.name) for s in index1.symbols] == [(s.file, s.name) for s in index2.symbols]


class TestFileInfo:
    """Tests for FileInfo dataclass."""
    
    def test_to_dict(self) -> None:
        """Test FileInfo serialization."""
        info = FileInfo(path="test.py", size=100, mtime=1234567890.0, language="python")
        
        d = info.to_dict()
        
        assert d["path"] == "test.py"
        assert d["size"] == 100
        assert d["language"] == "python"


class TestSymbolInfo:
    """Tests for SymbolInfo dataclass."""
    
    def test_to_dict(self) -> None:
        """Test SymbolInfo serialization."""
        info = SymbolInfo(name="my_func", kind="function", file="test.py", line=10)
        
        d = info.to_dict()
        
        assert d["name"] == "my_func"
        assert d["kind"] == "function"
        assert d["line"] == 10
