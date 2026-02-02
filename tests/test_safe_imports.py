"""Tests for safe imports without optional provider SDKs.

This module tests that core modules can be imported even when optional
LLM provider SDKs (google-genai, openai) are not installed.
"""

import builtins
import sys
from unittest.mock import patch

import pytest


class TestSafeImports:
    """Test that core modules import safely without provider SDKs."""

    def test_controller_imports_without_provider_sdks(self):
        """Controller module should import even if provider SDKs are missing.

        This test simulates a missing google.genai and openai by mocking
        the import to raise ImportError. The controller should still import
        successfully.
        """
        # Save original modules if they exist
        original_genai = sys.modules.get("google.genai")
        original_openai = sys.modules.get("openai")

        try:
            # Remove provider modules from sys.modules to simulate them not being installed
            if "google.genai" in sys.modules:
                del sys.modules["google.genai"]
            if "openai" in sys.modules:
                del sys.modules["openai"]

            # Reload llm modules to clear any cached imports
            if "rfsn_controller.llm_gemini" in sys.modules:
                del sys.modules["rfsn_controller.llm_gemini"]
            if "rfsn_controller.llm_deepseek" in sys.modules:
                del sys.modules["rfsn_controller.llm_deepseek"]
            if "rfsn_controller.controller" in sys.modules:
                del sys.modules["rfsn_controller.controller"]

            # Mock imports to raise ImportError
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name in ("google.genai", "openai", "google"):
                    raise ImportError(f"No module named '{name}'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                # This should NOT raise ImportError
                import rfsn_controller.controller

                # Verify the module imported successfully
                assert rfsn_controller.controller is not None

        finally:
            # Restore original modules
            if original_genai is not None:
                sys.modules["google.genai"] = original_genai
            if original_openai is not None:
                sys.modules["openai"] = original_openai

    def test_llm_gemini_raises_runtime_error_on_call_without_sdk(self):
        """llm_gemini should raise RuntimeError when called without SDK installed."""
        # Import the module (should work)
        import rfsn_controller.llm_gemini

        # Clear the SDK import cache
        rfsn_controller.llm_gemini._genai = None
        rfsn_controller.llm_gemini._types = None

        # Mock the import to fail
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "google.genai" in name or name == "google":
                raise ImportError("No module named 'google.genai'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Calling call_model should raise RuntimeError (not ImportError)
            with pytest.raises(RuntimeError, match="Google GenAI SDK not available"):
                rfsn_controller.llm_gemini.call_model("test")

    def test_llm_deepseek_raises_runtime_error_on_call_without_sdk(self):
        """llm_deepseek should raise RuntimeError when called without SDK installed."""
        # Import the module (should work)
        import rfsn_controller.llm_deepseek

        # Clear the SDK import cache
        rfsn_controller.llm_deepseek._openai = None

        # Mock the import to fail
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "openai" in name:
                raise ImportError("No module named 'openai'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            # Calling call_model should raise RuntimeError (not ImportError)
            with pytest.raises(RuntimeError, match="OpenAI SDK not available"):
                rfsn_controller.llm_deepseek.call_model("test")

    def test_command_normalizer_imports_safely(self):
        """Command normalizer should import without any provider dependencies."""
        # This should always work
        from rfsn_controller.command_normalizer import detect_shell_idioms

        # Test basic functionality
        assert callable(detect_shell_idioms)
        assert detect_shell_idioms("echo hello") is False
        assert detect_shell_idioms("echo hello && echo world") is True

    def test_sandbox_imports_safely(self):
        """Sandbox module should import without any provider dependencies."""
        from rfsn_controller.sandbox import Sandbox, create_sandbox

        assert callable(create_sandbox)
        assert Sandbox is not None

    def test_verifier_imports_safely(self):
        """Verifier module should import without any provider dependencies."""
        from rfsn_controller.verifier import Verifier, VerifyResult

        assert VerifyResult is not None
        assert Verifier is not None

    def test_patch_hygiene_imports_safely(self):
        """Patch hygiene module should import without any provider dependencies."""
        from rfsn_controller.patch_hygiene import PatchHygieneConfig, validate_patch_hygiene

        assert PatchHygieneConfig is not None
        assert callable(validate_patch_hygiene)
