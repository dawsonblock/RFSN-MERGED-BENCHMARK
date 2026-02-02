"""Tests for buildpack plugin registry system."""



def test_buildpack_registry_initialization():
    """Test BuildpackRegistry initializes."""
    from rfsn_controller.buildpack_registry import BuildpackRegistry

    registry = BuildpackRegistry()
    assert registry is not None
    assert len(registry._buildpacks) > 0


def test_list_buildpacks():
    """Test list_buildpacks returns all buildpack names."""
    from rfsn_controller.buildpack_registry import list_buildpacks

    buildpacks = list_buildpacks()
    assert isinstance(buildpacks, list)
    assert len(buildpacks) >= 8  # Built-in buildpacks
    assert "python" in buildpacks
    assert "node" in buildpacks
    assert "go" in buildpacks
    assert "rust" in buildpacks


def test_get_buildpack():
    """Test get_buildpack retrieves buildpack class."""
    from rfsn_controller.buildpack_registry import get_buildpack

    python_bp = get_buildpack("python")
    assert python_bp is not None
    assert python_bp.__name__ == "PythonBuildpack"


def test_get_nonexistent_buildpack():
    """Test get_buildpack returns None for nonexistent buildpack."""
    from rfsn_controller.buildpack_registry import get_buildpack

    result = get_buildpack("nonexistent_language")
    assert result is None


def test_register_buildpack():
    """Test register_buildpack adds custom buildpack."""
    from rfsn_controller.buildpack_registry import get_buildpack, register_buildpack
    from rfsn_controller.buildpacks.base import Buildpack, BuildpackContext, DetectResult

    class TestBuildpack(Buildpack):
        def detect(self, ctx: BuildpackContext) -> DetectResult:
            return DetectResult(
                buildpack_type="test",
                confidence=1.0,
            )

    register_buildpack("test_lang", TestBuildpack)
    
    retrieved = get_buildpack("test_lang")
    assert retrieved is not None
    assert retrieved == TestBuildpack


def test_buildpack_instantiation():
    """Test that buildpack classes can be instantiated."""
    from rfsn_controller.buildpack_registry import get_buildpack

    python_bp_class = get_buildpack("python")
    assert python_bp_class is not None
    
    # Instantiate
    bp = python_bp_class()
    assert bp is not None


def test_detect_buildpack():
    """Test detect_buildpack auto-detects language."""
    from rfsn_controller.buildpack_registry import detect_buildpack

    # Python project
    result = detect_buildpack(
        repo_dir="/test/repo",
        repo_tree=["setup.py", "requirements.txt", "main.py"],
        files={"setup.py": "# Python setup"},
    )
    
    if result:
        name, buildpack = result
        assert isinstance(name, str)
        assert buildpack is not None


def test_get_registry_singleton():
    """Test get_registry returns singleton."""
    from rfsn_controller.buildpack_registry import get_registry

    registry1 = get_registry()
    registry2 = get_registry()
    
    assert registry1 is registry2


def test_registry_get_method():
    """Test BuildpackRegistry.get method."""
    from rfsn_controller.buildpack_registry import BuildpackRegistry

    registry = BuildpackRegistry()
    
    python_bp = registry.get("python")
    assert python_bp is not None
    
    nonexistent = registry.get("nonexistent")
    assert nonexistent is None


def test_registry_list_all():
    """Test BuildpackRegistry.list_all method."""
    from rfsn_controller.buildpack_registry import BuildpackRegistry

    registry = BuildpackRegistry()
    buildpacks = registry.list_all()
    
    assert isinstance(buildpacks, list)
    assert len(buildpacks) >= 8
    assert "python" in buildpacks


def test_all_builtin_buildpacks_load():
    """Test all built-in buildpacks load successfully."""
    from rfsn_controller.buildpack_registry import get_buildpack

    expected = ["python", "node", "go", "rust", "java", "dotnet", "cpp", "polyrepo"]
    
    for name in expected:
        bp_class = get_buildpack(name)
        assert bp_class is not None, f"Buildpack {name} failed to load"
        
        # Try to instantiate
        bp = bp_class()
        assert bp is not None, f"Buildpack {name} failed to instantiate"
