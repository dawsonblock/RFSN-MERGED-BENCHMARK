"""
Staged Test Runner for RFSN

Executes tests in stages:
1. Pre-patch baseline (establish baseline failures)
2. Post-patch validation (verify patch fixes issue)
3. Full test suite (ensure no regressions)

Supports:
- Docker sandbox isolation
- Artifact capture (logs, traces, timing)
- Timeout handling
- Multiple test frameworks (pytest, unittest, etc.)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


class TestStage(str, Enum):
    """Test execution stages"""
    BASELINE = "baseline"           # Pre-patch tests
    VALIDATION = "validation"       # Post-patch targeted tests
    REGRESSION = "regression"       # Full test suite
    SMOKE = "smoke"                 # Quick smoke tests


class TestStatus(str, Enum):
    """Test execution status"""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    status: TestStatus
    duration_ms: float
    output: str = ""
    error: str = ""
    traceback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Results from a test stage"""
    stage: TestStage
    passed: int = 0
    failed: int = 0
    error: int = 0
    skipped: int = 0
    timeout: int = 0
    duration_ms: float = 0
    tests: List[TestResult] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestRunConfig:
    """Configuration for test execution"""
    repo_path: str
    test_command: str = "pytest"
    test_args: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    use_docker: bool = False
    docker_image: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    capture_artifacts: bool = True


class StagedTestRunner:
    """Execute tests in stages with artifact capture"""
    
    def __init__(self, config: TestRunConfig):
        self.config = config
        self.baseline_results: Optional[StageResult] = None
    
    async def run_baseline(self) -> StageResult:
        """
        Run baseline tests before applying patch
        
        Establishes which tests were already failing
        """
        logger.info("Running baseline tests...")
        
        result = await self._run_stage(
            stage=TestStage.BASELINE,
            test_filter=None
        )
        
        self.baseline_results = result
        
        logger.info(
            f"Baseline: {result.passed} passed, {result.failed} failed, "
            f"{result.error} errors in {result.duration_ms/1000:.1f}s"
        )
        
        return result
    
    async def run_validation(
        self,
        target_tests: Optional[List[str]] = None
    ) -> StageResult:
        """
        Run validation tests after applying patch
        
        Focuses on tests that should now pass
        """
        logger.info("Running validation tests...")
        
        result = await self._run_stage(
            stage=TestStage.VALIDATION,
            test_filter=target_tests
        )
        
        logger.info(
            f"Validation: {result.passed} passed, {result.failed} failed, "
            f"{result.error} errors in {result.duration_ms/1000:.1f}s"
        )
        
        return result
    
    async def run_regression(self) -> StageResult:
        """
        Run full test suite to check for regressions
        
        Ensures patch didn't break other tests
        """
        logger.info("Running regression tests...")
        
        result = await self._run_stage(
            stage=TestStage.REGRESSION,
            test_filter=None
        )
        
        logger.info(
            f"Regression: {result.passed} passed, {result.failed} failed, "
            f"{result.error} errors in {result.duration_ms/1000:.1f}s"
        )
        
        return result
    
    async def _run_stage(
        self,
        stage: TestStage,
        test_filter: Optional[List[str]] = None
    ) -> StageResult:
        """Execute a test stage"""
        
        start_time = time.time()
        
        # Build test command
        cmd = self._build_test_command(test_filter)
        
        # Execute tests
        if self.config.use_docker:
            result = await self._run_in_docker(cmd)
        else:
            result = await self._run_locally(cmd)
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Parse test results
        tests = self._parse_test_output(result["stdout"], result["stderr"])
        
        # Aggregate results
        stage_result = StageResult(
            stage=stage,
            duration_ms=duration_ms,
            tests=tests
        )
        
        for test in tests:
            if test.status == TestStatus.PASSED:
                stage_result.passed += 1
            elif test.status == TestStatus.FAILED:
                stage_result.failed += 1
            elif test.status == TestStatus.ERROR:
                stage_result.error += 1
            elif test.status == TestStatus.SKIPPED:
                stage_result.skipped += 1
            elif test.status == TestStatus.TIMEOUT:
                stage_result.timeout += 1
        
        # Capture artifacts
        if self.config.capture_artifacts:
            stage_result.artifacts = {
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": str(result["returncode"]),
                "command": " ".join(cmd)
            }
        
        return stage_result
    
    def _build_test_command(
        self,
        test_filter: Optional[List[str]] = None
    ) -> List[str]:
        """Build test command with arguments"""
        
        cmd = [self.config.test_command]
        
        # Add configured args
        cmd.extend(self.config.test_args)
        
        # Add test filter
        if test_filter:
            for test in test_filter:
                cmd.append(test)
        
        # Common pytest args for better output
        if "pytest" in self.config.test_command:
            if "-v" not in cmd:
                cmd.append("-v")
            if "--tb=short" not in cmd:
                cmd.append("--tb=short")
            cmd.append("--json-report")
            cmd.append("--json-report-file=test-report.json")
        
        return cmd
    
    async def _run_locally(self, cmd: List[str]) -> Dict[str, Any]:
        """Run tests locally"""
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.config.working_dir or self.config.repo_path,
                env={**os.environ, **self.config.env_vars},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout_seconds
                )
                
                return {
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "returncode": process.returncode
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return {
                    "stdout": "",
                    "stderr": f"Test execution timed out after {self.config.timeout_seconds}s",
                    "returncode": -1
                }
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    async def _run_in_docker(self, cmd: List[str]) -> Dict[str, Any]:
        """Run tests in Docker container"""
        
        if not self.config.docker_image:
            logger.warning("Docker image not specified, falling back to local execution")
            return await self._run_locally(cmd)
        
        # Build docker run command
        docker_cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{self.config.repo_path}:/workspace",
            "-w", "/workspace",
            f"--cpus={os.cpu_count() or 4}",
            "--memory=4g",
            "--network=none",  # Isolate network
        ]
        
        # Add environment variables
        for key, value in self.config.env_vars.items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        
        # Add image and command
        docker_cmd.append(self.config.docker_image)
        docker_cmd.extend(cmd)
        
        logger.info(f"Running in Docker: {' '.join(docker_cmd)}")
        
        return await self._run_locally(docker_cmd)
    
    def _parse_test_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse test output to extract individual test results"""
        
        tests = []
        
        # Try to parse pytest JSON report first
        json_report_path = Path(self.config.repo_path) / "test-report.json"
        if json_report_path.exists():
            try:
                with open(json_report_path) as f:
                    report = json.load(f)
                    tests.extend(self._parse_pytest_json(report))
                json_report_path.unlink()  # Clean up
                return tests
            except Exception as e:
                logger.warning(f"Failed to parse JSON report: {e}")
        
        # Fallback: parse from stdout
        tests.extend(self._parse_pytest_output(stdout))
        
        if not tests:
            # If no tests parsed, create a summary test
            tests.append(TestResult(
                test_id="test_suite",
                status=TestStatus.PASSED if "passed" in stdout.lower() else TestStatus.FAILED,
                duration_ms=0,
                output=stdout,
                error=stderr
            ))
        
        return tests
    
    def _parse_pytest_json(self, report: Dict[str, Any]) -> List[TestResult]:
        """Parse pytest JSON report"""
        
        tests = []
        
        for test in report.get("tests", []):
            status_map = {
                "passed": TestStatus.PASSED,
                "failed": TestStatus.FAILED,
                "error": TestStatus.ERROR,
                "skipped": TestStatus.SKIPPED
            }
            
            tests.append(TestResult(
                test_id=test.get("nodeid", "unknown"),
                status=status_map.get(test.get("outcome", "failed"), TestStatus.FAILED),
                duration_ms=test.get("duration", 0) * 1000,
                output=test.get("call", {}).get("stdout", ""),
                error=test.get("call", {}).get("stderr", ""),
                traceback="\n".join(test.get("call", {}).get("longrepr", [])),
                metadata=test
            ))
        
        return tests
    
    def _parse_pytest_output(self, output: str) -> List[TestResult]:
        """Parse pytest verbose output"""
        
        tests = []
        
        for line in output.split('\n'):
            # Look for test results: test_name.py::test_function PASSED
            if '::' in line and any(status in line for status in ['PASSED', 'FAILED', 'ERROR', 'SKIPPED']):
                parts = line.split()
                if len(parts) >= 2:
                    test_id = parts[0]
                    status_str = parts[-1].lower()
                    
                    status_map = {
                        'passed': TestStatus.PASSED,
                        'failed': TestStatus.FAILED,
                        'error': TestStatus.ERROR,
                        'skipped': TestStatus.SKIPPED
                    }
                    
                    tests.append(TestResult(
                        test_id=test_id,
                        status=status_map.get(status_str, TestStatus.FAILED),
                        duration_ms=0,
                        output=line
                    ))
        
        return tests
    
    def compare_with_baseline(self, validation: StageResult) -> Dict[str, Any]:
        """
        Compare validation results with baseline
        
        Returns analysis of improvements and regressions
        """
        if not self.baseline_results:
            logger.warning("No baseline results to compare against")
            return {}
        
        baseline_failed = {t.test_id for t in self.baseline_results.tests if t.status == TestStatus.FAILED}
        validation_failed = {t.test_id for t in validation.tests if t.status == TestStatus.FAILED}
        
        baseline_passed = {t.test_id for t in self.baseline_results.tests if t.status == TestStatus.PASSED}
        validation_passed = {t.test_id for t in validation.tests if t.status == TestStatus.PASSED}
        
        # Tests that were failing but now pass
        fixed = baseline_failed - validation_failed
        
        # Tests that were passing but now fail (regressions!)
        regressions = baseline_passed & validation_failed
        
        # Tests still failing
        still_failing = baseline_failed & validation_failed
        
        analysis = {
            "fixed_tests": list(fixed),
            "regression_tests": list(regressions),
            "still_failing": list(still_failing),
            "summary": {
                "fixed_count": len(fixed),
                "regression_count": len(regressions),
                "still_failing_count": len(still_failing),
                "improvement": len(fixed) > 0 and len(regressions) == 0
            }
        }
        
        logger.info(
            f"Comparison: {len(fixed)} fixed, {len(regressions)} regressions, "
            f"{len(still_failing)} still failing"
        )
        
        return analysis


async def run_staged_tests(
    repo_path: str,
    test_command: str = "pytest",
    use_docker: bool = False,
    docker_image: Optional[str] = None,
    target_tests: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    High-level function to run staged tests
    
    Returns complete test results with comparison
    """
    
    config = TestRunConfig(
        repo_path=repo_path,
        test_command=test_command,
        use_docker=use_docker,
        docker_image=docker_image
    )
    
    runner = StagedTestRunner(config)
    
    # Run baseline
    baseline = await runner.run_baseline()
    
    # Run validation (after patch would be applied)
    validation = await runner.run_validation(target_tests)
    
    # Compare results
    comparison = runner.compare_with_baseline(validation)
    
    return {
        "baseline": baseline,
        "validation": validation,
        "comparison": comparison,
        "success": comparison.get("summary", {}).get("improvement", False)
    }


if __name__ == "__main__":
    # Test the runner
    async def test():
        results = await run_staged_tests(
            repo_path=".",
            test_command="pytest",
            target_tests=["tests/"]
        )
        
        print(f"\n✅ Baseline: {results['baseline'].passed} passed, {results['baseline'].failed} failed")
        print(f"✅ Validation: {results['validation'].passed} passed, {results['validation'].failed} failed")
        print(f"✅ Fixed: {len(results['comparison']['fixed_tests'])}")
        print(f"❌ Regressions: {len(results['comparison']['regression_tests'])}")
    
    asyncio.run(test())
