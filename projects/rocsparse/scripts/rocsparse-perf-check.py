#!/usr/bin/env python3
"""
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Performance regression testing script for rocSPARSE.

This script:
1. Runs benchmarks for a subset of routines (spmv, spmm, spsv) with specified matrices
2. Compares results against a multi-architecture baseline file
3. Reports performance regressions based on configurable tolerance
4. Can run locally or in CI/CD pipelines

Usage:
    # Run regression test
    ./rocsparse-perf-check.py --config perf-test-config.yaml
    
    # Update baseline after performance improvements
    ./rocsparse-perf-check.py --config perf-test-config.yaml --update-baseline
    
    # Verbose output
    ./rocsparse-perf-check.py --config perf-test-config.yaml -v
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run"""
    routine: str
    precision: str
    matrix_path: str
    alpha: float = 1.0
    beta: float = 0.0
    iters: int = 1000
    extra_args: List[str] = None


@dataclass
class TestResult:
    """Result from a benchmark execution"""
    routine: str
    precision: str
    matrix: str
    gflops: float
    bandwidth_gb: float
    time_ms: float
    architecture: str


class PerformanceChecker:
    """Main class for performance regression checking"""
    
    def __init__(self, config_file: str, verbose: bool = False):
        self.config_file = config_file
        self.verbose = verbose
        self.config = self._load_config()
        self.architecture = self._detect_architecture()
        self.rocsparse_bench = self._find_rocsparse_bench()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required = ['baseline_file', 'routines', 'matrices', 'tolerance']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        return config
    
    def _detect_architecture(self) -> str:
        """Detect the current GPU architecture"""
        try:
            # Try rocminfo to detect GPU
            result = subprocess.run(['rocminfo'], capture_output=True, text=True)
            output = result.stdout
            
            # Look for gfx architecture
            gfx_match = re.search(r'gfx(\d+)', output)
            if gfx_match:
                gfx_code = gfx_match.group(1)
                
                # Map to friendly names
                arch_map = {
                    '90a': 'mi300',
                    '908': 'mi250',
                    '942': 'mi300',
                    '1100': 'navi31',
                    '1101': 'navi32',
                    '1102': 'navi33',
                    '1103': 'navi4',
                }
                
                arch = arch_map.get(gfx_code, f'gfx{gfx_code}')
                if self.verbose:
                    print(f"Detected architecture: {arch} (gfx{gfx_code})")
                return arch
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not detect architecture: {e}")
        
        # Fallback to environment variable or config
        if 'ROCSPARSE_TEST_ARCH' in os.environ:
            return os.environ['ROCSPARSE_TEST_ARCH']
        
        if 'default_architecture' in self.config:
            return self.config['default_architecture']
        
        print("Warning: Could not detect architecture, using 'unknown'")
        return 'unknown'
    
    def _find_rocsparse_bench(self) -> str:
        """Find the rocsparse-bench executable"""
        # Check config
        if 'rocsparse_bench_path' in self.config:
            bench_path = self.config['rocsparse_bench_path']
            if os.path.exists(bench_path):
                return bench_path
        
        # Check environment variable
        if 'ROCSPARSE_BENCH' in os.environ:
            bench_path = os.environ['ROCSPARSE_BENCH']
            if os.path.exists(bench_path):
                return bench_path
        
        # Search in common locations
        search_paths = [
            './build/release/clients/staging/rocsparse-bench',
            './build/debug/clients/staging/rocsparse-bench',
            '../build/release/clients/staging/rocsparse-bench',
            '/opt/rocm/bin/rocsparse-bench',
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                if self.verbose:
                    print(f"Found rocsparse-bench at: {path}")
                return path
        
        raise FileNotFoundError(
            "Could not find rocsparse-bench executable. "
            "Please specify 'rocsparse_bench_path' in config or set ROCSPARSE_BENCH env var."
        )
    
    def _expand_matrix_path(self, matrix_spec: str) -> List[str]:
        """Expand matrix specification (supports wildcards and directories)"""
        import glob
        
        # Handle environment variable substitution
        if '$ROCSPARSE_BENCH_DATA_DIR' in matrix_spec:
            data_dir = os.environ.get('ROCSPARSE_BENCH_DATA_DIR', '')
            if not data_dir:
                print(f"Warning: ROCSPARSE_BENCH_DATA_DIR not set, skipping {matrix_spec}")
                return []
            matrix_spec = matrix_spec.replace('$ROCSPARSE_BENCH_DATA_DIR', data_dir)
        
        # Expand wildcards
        expanded = glob.glob(matrix_spec)
        if not expanded:
            print(f"Warning: No matrices found matching: {matrix_spec}")
            return []
        
        return expanded
    
    def _run_benchmark(self, bench_config: BenchmarkConfig) -> Optional[TestResult]:
        """Run a single benchmark and return results"""
        cmd = [
            self.rocsparse_bench,
            '-f', bench_config.routine,
            '--precision', bench_config.precision,
            '--alpha', str(bench_config.alpha),
            '--beta', str(bench_config.beta),
            '--iters', str(bench_config.iters),
        ]
        
        # Add matrix file
        matrix_ext = Path(bench_config.matrix_path).suffix
        if matrix_ext == '.csr':
            cmd.extend(['--rocalution', bench_config.matrix_path])
        elif matrix_ext == '.mtx':
            cmd.extend(['--mtx', bench_config.matrix_path])
        else:
            cmd.extend(['--rocalution', bench_config.matrix_path])
        
        # Add extra arguments if any
        if bench_config.extra_args:
            cmd.extend(bench_config.extra_args)
        
        # Add JSON output
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        cmd.extend(['--bench-o', output_file])
        
        if self.verbose:
            print(f"\nRunning: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Error running benchmark: {result.stderr}")
                return None
            
            # Parse JSON output
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Extract results (assuming single test case)
            if not data or len(data) == 0:
                print(f"Warning: No results in output file")
                return None
            
            result_data = data[0]
            matrix_name = Path(bench_config.matrix_path).stem
            
            return TestResult(
                routine=bench_config.routine,
                precision=bench_config.precision,
                matrix=matrix_name,
                gflops=float(result_data.get('GFlops/s', 0)),
                bandwidth_gb=float(result_data.get('GB/s', 0)),
                time_ms=float(result_data.get('msec', 0)),
                architecture=self.architecture
            )
            
        except subprocess.TimeoutExpired:
            print(f"Timeout running benchmark for {bench_config.routine}")
            return None
        except Exception as e:
            print(f"Error running benchmark: {e}")
            return None
        finally:
            # Cleanup temp file
            if os.path.exists(output_file):
                os.remove(output_file)
    
    def run_all_benchmarks(self) -> List[TestResult]:
        """Run all configured benchmarks"""
        results = []
        
        routines = self.config['routines']
        matrices = self.config['matrices']
        
        print(f"\n{'='*60}")
        print(f"Running benchmarks for architecture: {self.architecture}")
        print(f"{'='*60}")
        
        total_tests = 0
        for routine_config in routines:
            routine = routine_config['name']
            precisions = routine_config.get('precisions', ['d'])
            extra_args = routine_config.get('extra_args', [])
            
            for matrix_spec in matrices:
                matrix_files = self._expand_matrix_path(matrix_spec)
                for matrix_file in matrix_files:
                    for precision in precisions:
                        total_tests += 1
        
        current_test = 0
        for routine_config in routines:
            routine = routine_config['name']
            precisions = routine_config.get('precisions', ['d'])
            extra_args = routine_config.get('extra_args', [])
            
            for matrix_spec in matrices:
                matrix_files = self._expand_matrix_path(matrix_spec)
                
                for matrix_file in matrix_files:
                    for precision in precisions:
                        current_test += 1
                        
                        print(f"\n[{current_test}/{total_tests}] Testing {routine} "
                              f"(precision={precision}) on {Path(matrix_file).name}")
                        
                        bench_config = BenchmarkConfig(
                            routine=routine,
                            precision=precision,
                            matrix_path=matrix_file,
                            extra_args=extra_args
                        )
                        
                        result = self._run_benchmark(bench_config)
                        if result:
                            results.append(result)
                            if self.verbose:
                                print(f"  GFlops: {result.gflops:.2f}, "
                                      f"Bandwidth: {result.bandwidth_gb:.2f} GB/s, "
                                      f"Time: {result.time_ms:.4f} ms")
        
        print(f"\n{'='*60}")
        print(f"Completed {len(results)}/{total_tests} benchmarks successfully")
        print(f"{'='*60}\n")
        
        return results
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline results from file"""
        baseline_file = self.config['baseline_file']
        
        if not os.path.exists(baseline_file):
            print(f"Warning: Baseline file not found: {baseline_file}")
            return {}
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
    
    def _save_baseline(self, baseline: Dict[str, Any]):
        """Save baseline results to file"""
        baseline_file = self.config['baseline_file']
        
        # Create directory if needed
        os.makedirs(os.path.dirname(baseline_file) or '.', exist_ok=True)
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2, sort_keys=True)
        
        print(f"Baseline file updated: {baseline_file}")
    
    def _get_baseline_key(self, result: TestResult) -> str:
        """Generate key for baseline lookup"""
        return f"{result.architecture}:{result.routine}:{result.precision}:{result.matrix}"
    
    def compare_with_baseline(self, results: List[TestResult]) -> Tuple[bool, List[str]]:
        """Compare results with baseline and return (passed, messages)"""
        baseline = self._load_baseline()
        
        if not baseline:
            print("\nNo baseline found. Run with --update-baseline to create one.")
            return False, ["No baseline file"]
        
        tolerance = self.config['tolerance']
        passed = True
        messages = []
        
        print(f"\n{'='*60}")
        print(f"Comparing against baseline (tolerance: {tolerance}%)")
        print(f"{'='*60}\n")
        
        # Check for architecture in baseline
        if self.architecture not in baseline.get('architectures', {}):
            msg = f"Warning: Architecture '{self.architecture}' not found in baseline"
            print(msg)
            messages.append(msg)
            return False, messages
        
        arch_baseline = baseline['architectures'][self.architecture]
        
        for result in results:
            key = f"{result.routine}:{result.precision}:{result.matrix}"
            
            if key not in arch_baseline:
                msg = f"⚠️  {key}: Not in baseline (skipping)"
                print(msg)
                messages.append(msg)
                continue
            
            baseline_data = arch_baseline[key]
            baseline_gflops = baseline_data['gflops']
            baseline_bandwidth = baseline_data['bandwidth_gb']
            baseline_time = baseline_data['time_ms']
            
            # Calculate percentage differences
            gflops_diff = ((result.gflops - baseline_gflops) / baseline_gflops) * 100
            bandwidth_diff = ((result.bandwidth_gb - baseline_bandwidth) / baseline_bandwidth) * 100
            time_diff = ((result.time_ms - baseline_time) / baseline_time) * 100
            
            # Check if any metric regressed beyond tolerance
            regression = False
            status = "✅"
            details = []
            
            if gflops_diff < -tolerance:
                regression = True
                status = "❌"
                details.append(f"GFlops: {gflops_diff:+.2f}%")
            
            if bandwidth_diff < -tolerance:
                regression = True
                status = "❌"
                details.append(f"Bandwidth: {bandwidth_diff:+.2f}%")
            
            if time_diff > tolerance:  # Higher time is worse
                regression = True
                status = "❌"
                details.append(f"Time: {time_diff:+.2f}%")
            
            # Format output
            msg = f"{status} {key}"
            if regression:
                msg += f" - REGRESSION: {', '.join(details)}"
                passed = False
            else:
                msg += f" - OK (GFlops: {gflops_diff:+.2f}%, Time: {time_diff:+.2f}%)"
            
            print(msg)
            messages.append(msg)
        
        print(f"\n{'='*60}")
        if passed:
            print("✅ All tests PASSED - No performance regressions detected")
        else:
            print("❌ Some tests FAILED - Performance regressions detected")
        print(f"{'='*60}\n")
        
        return passed, messages
    
    def update_baseline(self, results: List[TestResult]):
        """Update baseline with new results"""
        baseline = self._load_baseline()
        
        # Initialize structure if needed
        if 'version' not in baseline:
            baseline['version'] = '1.0'
        
        if 'architectures' not in baseline:
            baseline['architectures'] = {}
        
        if self.architecture not in baseline['architectures']:
            baseline['architectures'][self.architecture] = {}
        
        arch_baseline = baseline['architectures'][self.architecture]
        
        # Update results
        updated_count = 0
        new_count = 0
        
        for result in results:
            key = f"{result.routine}:{result.precision}:{result.matrix}"
            
            is_new = key not in arch_baseline
            
            arch_baseline[key] = {
                'gflops': result.gflops,
                'bandwidth_gb': result.bandwidth_gb,
                'time_ms': result.time_ms,
                'routine': result.routine,
                'precision': result.precision,
                'matrix': result.matrix
            }
            
            if is_new:
                new_count += 1
            else:
                updated_count += 1
        
        # Add metadata
        import datetime
        baseline['last_updated'] = datetime.datetime.now().isoformat()
        baseline['last_updated_arch'] = self.architecture
        
        self._save_baseline(baseline)
        
        print(f"\nBaseline updated:")
        print(f"  Architecture: {self.architecture}")
        print(f"  New entries: {new_count}")
        print(f"  Updated entries: {updated_count}")


def main():
    parser = argparse.ArgumentParser(
        description='rocSPARSE performance regression testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run regression test
  ./rocsparse-perf-check.py --config perf-test-config.yaml
  
  # Update baseline after verifying improvements
  ./rocsparse-perf-check.py --config perf-test-config.yaml --update-baseline
  
  # Verbose output
  ./rocsparse-perf-check.py --config perf-test-config.yaml -v
        """
    )
    
    parser.add_argument('--config', '-c', required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--update-baseline', '-u', action='store_true',
                        help='Update baseline file with current results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        checker = PerformanceChecker(args.config, args.verbose)
        results = checker.run_all_benchmarks()
        
        if not results:
            print("Error: No benchmark results obtained")
            return 1
        
        if args.update_baseline:
            checker.update_baseline(results)
            return 0
        else:
            passed, messages = checker.compare_with_baseline(results)
            return 0 if passed else 1
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

