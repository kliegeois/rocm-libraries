#!/usr/bin/env python3
"""
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Performance regression testing script for rocSPARSE with statistical analysis.

This script:
1. Runs benchmarks multiple times for statistical robustness
2. Calculates confidence intervals for performance metrics
3. Compares results against a multi-architecture baseline file
4. Reports performance regressions based on statistical significance
5. Can run locally or in CI/CD pipelines

Usage:
    # Run regression test
    ./rocsparse-perf-check.py --config perf-test-config.yaml
    
    # Update baseline after performance improvements
    ./rocsparse-perf-check.py --config perf-test-config.yaml --update-baseline
    
    # Verbose output
    ./rocsparse-perf-check.py --config perf-test-config.yaml -v
    
    # Run with custom number of iterations and confidence level
    ./rocsparse-perf-check.py --config perf-test-config.yaml --num-runs 10 --confidence 0.99
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import statistics
import math


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
class SingleRunResult:
    """Result from a single benchmark execution"""
    gflops: float
    bandwidth_gb: float
    time_ms: float


@dataclass
class StatisticalResult:
    """Statistical summary of multiple benchmark runs"""
    routine: str
    precision: str
    matrix: str
    architecture: str
    
    # GFlops statistics
    gflops_mean: float
    gflops_median: float
    gflops_std: float
    gflops_ci_lower: float
    gflops_ci_upper: float
    
    # Bandwidth statistics
    bandwidth_mean: float
    bandwidth_median: float
    bandwidth_std: float
    bandwidth_ci_lower: float
    bandwidth_ci_upper: float
    
    # Time statistics
    time_mean: float
    time_median: float
    time_std: float
    time_ci_lower: float
    time_ci_upper: float
    
    # Metadata
    num_runs: int
    confidence_level: float
    
    @property
    def gflops_coefficient_of_variation(self) -> float:
        """Coefficient of variation for GFlops (std/mean)"""
        return (self.gflops_std / self.gflops_mean * 100) if self.gflops_mean > 0 else 0
    
    @property
    def time_coefficient_of_variation(self) -> float:
        """Coefficient of variation for time (std/mean)"""
        return (self.time_std / self.time_mean * 100) if self.time_mean > 0 else 0


class StatisticalAnalyzer:
    """Handles statistical analysis of benchmark results"""
    
    @staticmethod
    def calculate_confidence_interval(
        data: List[float], 
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate mean, median, std, and confidence interval for data.
        
        Returns: (mean, median, std, ci_lower, ci_upper)
        """
        if not data:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        if len(data) == 1:
            val = data[0]
            return val, val, 0.0, val, val
        
        mean = statistics.mean(data)
        median = statistics.median(data)
        std = statistics.stdev(data)
        n = len(data)
        
        # Use t-distribution for small sample sizes
        # For large n, t-distribution approaches normal distribution
        # Using approximation:  z_critical for n > 30critical 
        if n <= 30:
            # t-distribution critical values (two-tailed)
            # Approximation for common confidence levels
            t_critical_values = {
                0.90: {5: 2.015, 10: 1.812, 15: 1.753, 20: 1.725, 30: 1.697},
                0.95: {5: 2.571, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042},
                0.99: {5: 4.032, 10: 3.169, 15: 2.947, 20: 2.845, 30: 2.750},
            }
            
            # Find closest n in table
            available_n = [5, 10, 15, 20, 30]
            closest_n = min(available_n, key=lambda x: abs(x - n))
            
            t_critical = t_critical_values.get(confidence_level, {}).get(closest_n, 2.0)
        else:
            # For larger samples, use z-scores
            z_critical_values = {
                0.90: 1.645,
                0.95: 1.960,
                0.99: 2.576,
            }
            t_critical = z_critical_values.get(confidence_level, 1.960)
        
        # Calculate margin of error
        margin_of_error = t_critical * (std / math.sqrt(n))
        
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        return mean, median, std, ci_lower, ci_upper
    
    @staticmethod
    def analyze_runs(
        runs: List[SingleRunResult],
        routine: str,
        precision: str,
        matrix: str,
        architecture: str,
        confidence_level: float = 0.95
    ) -> StatisticalResult:
        """Analyze multiple runs and return statistical summary"""
        
        gflops_data = [r.gflops for r in runs]
        bandwidth_data = [r.bandwidth_gb for r in runs]
        time_data = [r.time_ms for r in runs]
        
        gflops_mean, gflops_median, gflops_std, gflops_ci_lower, gflops_ci_upper = \
            StatisticalAnalyzer.calculate_confidence_interval(gflops_data, confidence_level)
        
        bandwidth_mean, bandwidth_median, bandwidth_std, bandwidth_ci_lower, bandwidth_ci_upper = \
            StatisticalAnalyzer.calculate_confidence_interval(bandwidth_data, confidence_level)
        
        time_mean, time_median, time_std, time_ci_lower, time_ci_upper = \
            StatisticalAnalyzer.calculate_confidence_interval(time_data, confidence_level)
        
        return StatisticalResult(
            routine=routine,
            precision=precision,
            matrix=matrix,
            architecture=architecture,
            gflops_mean=gflops_mean,
            gflops_median=gflops_median,
            gflops_std=gflops_std,
            gflops_ci_lower=gflops_ci_lower,
            gflops_ci_upper=gflops_ci_upper,
            bandwidth_mean=bandwidth_mean,
            bandwidth_median=bandwidth_median,
            bandwidth_std=bandwidth_std,
            bandwidth_ci_lower=bandwidth_ci_lower,
            bandwidth_ci_upper=bandwidth_ci_upper,
            time_mean=time_mean,
            time_median=time_median,
            time_std=time_std,
            time_ci_lower=time_ci_lower,
            time_ci_upper=time_ci_upper,
            num_runs=len(runs),
            confidence_level=confidence_level
        )


class PerformanceChecker:
    """Main class for performance regression checking"""
    
    def __init__(self, config_file: str, verbose: bool = False, 
                 num_runs: int = None, confidence_level: float = None):
        self.config_file = config_file
        self.verbose = verbose
        self.config = self._load_config()
        
        # Override config with command-line arguments if provided
        self.num_runs = num_runs if num_runs is not None else self.config.get('num_runs', 5)
        self.confidence_level = confidence_level if confidence_level is not None else \
                               self.config.get('confidence_level', 0.95)
        
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
    
    def _run_single_benchmark(self, bench_config: BenchmarkConfig) -> Optional[SingleRunResult]:
        """Run a single benchmark iteration and return results"""
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
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"    Error running benchmark (exit code {result.returncode})")
                    print(f"      stderr: {result.stderr[:200]}")
                return None
            
            # Parse the tabular output from stdout
            output = result.stdout
            
            # Look for the data line
            lines = output.strip().split('\n')
            data_line = None
            for line in lines:
                if 'transA' in line or '---' in line or line.strip() == '':
                    continue
                if line.strip().startswith(('NT', 'T ')):
                    data_line = line
                    break
            
            if not data_line:
                if self.verbose:
                    print(f"    Error: Could not find result data in output")
                return None
            
            # Split the line by whitespace
            parts = data_line.split()
            
            if len(parts) < 12:
                if self.verbose:
                    print(f"    Error: Unexpected output format, got {len(parts)} fields")
                return None
            
            # Extract metrics
            try:
                gflops = float(parts[7])
                bandwidth_gb = float(parts[8])
                time_ms = float(parts[9])
            except (ValueError, IndexError) as e:
                if self.verbose:
                    print(f"    Error parsing benchmark results: {e}")
                return None
            
            return SingleRunResult(
                gflops=gflops,
                bandwidth_gb=bandwidth_gb,
                time_ms=time_ms
            )
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"    Timeout running benchmark")
            return None
        except Exception as e:
            if self.verbose:
                print(f"    Error running benchmark: {e}")
            return None
    
    def _run_benchmark_multiple_times(
        self, 
        bench_config: BenchmarkConfig
    ) -> Optional[StatisticalResult]:
        """Run benchmark multiple times and return statistical analysis"""
        
        print(f"    Running {self.num_runs} iterations...", end='', flush=True)
        
        runs = []
        for i in range(self.num_runs):
            result = self._run_single_benchmark(bench_config)
            if result:
                runs.append(result)
                if self.verbose:
                    print(f"\n      Run {i+1}/{self.num_runs}: "
                          f"GFlops={result.gflops:.2f}, "
                          f"Time={result.time_ms:.4f}ms", end='')
                else:
                    print('.', end='', flush=True)
            else:
                print('x', end='', flush=True)
        
        print()  # New line after progress indicators
        
        if not runs:
            print(f"    ❌ All runs failed")
            return None
        
        if len(runs) < self.num_runs:
            print(f"    ⚠️  Only {len(runs)}/{self.num_runs} runs succeeded")
        
        # Perform statistical analysis
        matrix_name = Path(bench_config.matrix_path).stem
        result = StatisticalAnalyzer.analyze_runs(
            runs=runs,
            routine=bench_config.routine,
            precision=bench_config.precision,
            matrix=matrix_name,
            architecture=self.architecture,
            confidence_level=self.confidence_level
        )
        
        if self.verbose or len(runs) < self.num_runs:
            cv = result.gflops_coefficient_of_variation
            print(f"    📊 Statistics: "
                  f"GFlops={result.gflops_mean:.2f}±{result.gflops_std:.2f} "
                  f"(CV={cv:.2f}%), "
                  f"CI=[{result.gflops_ci_lower:.2f}, {result.gflops_ci_upper:.2f}]")
        
        return result
    
    def run_all_benchmarks(self) -> List[StatisticalResult]:
        """Run all configured benchmarks with statistical analysis"""
        results = []
        
        routines = self.config['routines']
        matrices = self.config['matrices']
        
        print(f"\n{'='*70}")
        print(f"Running benchmarks for architecture: {self.architecture}")
        print(f"Statistical parameters: {self.num_runs} runs, "
              f"{self.confidence_level*100:.0f}% confidence interval")
        print(f"{'='*70}")
        
        total_tests = 0
        for routine_config in routines:
            routine = routine_config['name']
            precisions = routine_config.get('precisions', ['d'])
            
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
                        
                        result = self._run_benchmark_multiple_times(bench_config)
                        if result:
                            results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Completed {len(results)}/{total_tests} benchmarks successfully")
        print(f"{'='*70}\n")
        
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
    
    def compare_with_baseline(
        self, 
        results: List[StatisticalResult]
    ) -> Tuple[bool, List[str]]:
        """
        Compare results with baseline using statistical significance.
        
        Uses confidence intervals to determine if performance difference
        is statistically significant.
        """
        baseline = self._load_baseline()
        
        if not baseline:
            print("\nNo baseline found. Run with --update-baseline to create one.")
            return False, ["No baseline file"]
        
        tolerance = self.config['tolerance']
        passed = True
        messages = []
        
        print(f"\n{'='*70}")
        print(f"Comparing against baseline")
        print(f"  Tolerance: {tolerance}%")
        print(f"  Confidence level: {self.confidence_level*100:.0f}%")
        print(f"  Method: Statistical confidence interval overlap")
        print(f"{'='*70}\n")
        
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
            
            # Get baseline statistics
            baseline_gflops_mean = baseline_data.get('gflops_mean', baseline_data.get('gflops', 0))
            baseline_gflops_ci_lower = baseline_data.get('gflops_ci_lower', baseline_gflops_mean)
            baseline_gflops_ci_upper = baseline_data.get('gflops_ci_upper', baseline_gflops_mean)
            
            baseline_time_mean = baseline_data.get('time_mean', baseline_data.get('time_ms', 0))
            baseline_time_ci_lower = baseline_data.get('time_ci_lower', baseline_time_mean)
            baseline_time_ci_upper = baseline_data.get('time_ci_upper', baseline_time_mean)
            
            # Calculate percentage differences (based on means)
            gflops_diff = ((result.gflops_mean - baseline_gflops_mean) / 
                          baseline_gflops_mean) * 100 if baseline_gflops_mean > 0 else 0
            
            time_diff = ((result.time_mean - baseline_time_mean) / 
                        baseline_time_mean) * 100 if baseline_time_mean > 0 else 0
            
            # Check for statistical significance using confidence interval overlap
            # If CIs don't overlap, the difference is statistically significant
            gflops_ci_overlap = not (result.gflops_ci_upper < baseline_gflops_ci_lower or 
                                    result.gflops_ci_lower > baseline_gflops_ci_upper)
            
            time_ci_overlap = not (result.time_ci_upper < baseline_time_ci_lower or 
                                  result.time_ci_lower > baseline_time_ci_upper)
            
            # Determine if there's a regression
            regression = False
            status = "✅"
            details = []
            
            # GFlops regression: statistically significant decrease beyond tolerance
            if not gflops_ci_overlap and gflops_diff < -tolerance:
                regression = True
                details.append(f"GFlops: {gflops_diff:+.2f}% (significant)")
            elif gflops_diff < -tolerance:
                # Within statistical noise but shows concerning trend
                details.append(f"GFlops: {gflops_diff:+.2f}% (borderline)")
            
            # Time regression: statistically significant increase beyond tolerance
            if not time_ci_overlap and time_diff > tolerance:
                regression = True
                details.append(f"Time: {time_diff:+.2f}% (significant)")
            elif time_diff > tolerance:
                details.append(f"Time: {time_diff:+.2f}% (borderline)")
            
            if regression:
                status = "❌"
                passed = False
            elif details:
                status = "⚠️ "
            
            # Format output
            msg = f"{status} {key}"
            if details:
                msg += f" - {', '.join(details)}"
                if self.verbose:
                    msg += f"\n    Current:  GFlops={result.gflops_mean:.2f} " \
                           f"CI=[{result.gflops_ci_lower:.2f}, {result.gflops_ci_upper:.2f}], " \
                           f"CV={result.gflops_coefficient_of_variation:.2f}%"
                    msg += f"\n    Baseline: GFlops={baseline_gflops_mean:.2f} " \
                           f"CI=[{baseline_gflops_ci_lower:.2f}, {baseline_gflops_ci_upper:.2f}]"
            else:
                msg += f" - OK (GFlops: {gflops_diff:+.2f}%, Time: {time_diff:+.2f}%)"
            
            print(msg)
            messages.append(msg)
        
        print(f"\n{'='*70}")
        if passed:
            print("✅ All tests PASSED - No statistically significant regressions")
        else:
            print("❌ Some tests FAILED - Statistically significant regressions detected")
        print(f"{'='*70}\n")
        
        return passed, messages
    
    def update_baseline(self, results: List[StatisticalResult]):
        """Update baseline with new statistical results"""
        baseline = self._load_baseline()
        
        # Initialize structure if needed
        if 'version' not in baseline:
            baseline['version'] = '2.0'  # Version 2.0 includes statistical data
        
        if 'meta' not in baseline:
            baseline['meta'] = {}
        
        baseline['meta']['num_runs'] = self.num_runs
        baseline['meta']['confidence_level'] = self.confidence_level
        
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
                'routine': result.routine,
                'precision': result.precision,
                'matrix': result.matrix,
                # GFlops statistics
                'gflops_mean': result.gflops_mean,
                'gflops_median': result.gflops_median,
                'gflops_std': result.gflops_std,
                'gflops_ci_lower': result.gflops_ci_lower,
                'gflops_ci_upper': result.gflops_ci_upper,
                'gflops_cv': result.gflops_coefficient_of_variation,
                # Bandwidth statistics
                'bandwidth_mean': result.bandwidth_mean,
                'bandwidth_median': result.bandwidth_median,
                'bandwidth_std': result.bandwidth_std,
                'bandwidth_ci_lower': result.bandwidth_ci_lower,
                'bandwidth_ci_upper': result.bandwidth_ci_upper,
                # Time statistics
                'time_mean': result.time_mean,
                'time_median': result.time_median,
                'time_std': result.time_std,
                'time_ci_lower': result.time_ci_lower,
                'time_ci_upper': result.time_ci_upper,
                'time_cv': result.time_coefficient_of_variation,
                # Metadata
                'num_runs': result.num_runs,
                'confidence_level': result.confidence_level,
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
        print(f"  Statistical parameters: {self.num_runs} runs, "
              f"{self.confidence_level*100:.0f}% confidence")
        print(f"  New entries: {new_count}")
        print(f"  Updated entries: {updated_count}")


def main():
    parser = argparse.ArgumentParser(
        description='rocSPARSE performance regression testing with statistical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run regression test with default settings (5 runs, 95% confidence)
  ./rocsparse-perf-check.py --config perf-test-config.yaml
  
  # Run with higher statistical confidence (10 runs, 99% confidence)
  ./rocsparse-perf-check.py --config perf-test-config.yaml --num-runs 10 --confidence 0.99
  
  # Update baseline after verifying improvements
  ./rocsparse-perf-check.py --config perf-test-config.yaml --update-baseline
  
  # Verbose output with detailed statistics
  ./rocsparse-perf-check.py --config perf-test-config.yaml -v
  
Statistical Analysis:
  The script runs each benchmark multiple times and uses confidence intervals
  to determine if performance differences are statistically significant.
  A regression is reported only if:
    1. The performance difference exceeds the tolerance threshold, AND
    2. The confidence intervals don't overlap (statistically significant)
        """
    )
    
    parser.add_argument('--config', '-c', required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--update-baseline', '-u', action='store_true',
                        help='Update baseline file with current results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output with detailed statistics')
    parser.add_argument('--num-runs', '-n', type=int, default=None,
                        help='Number of runs per benchmark (default: 5 or from config)')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Confidence level for intervals: 0.90, 0.95, or 0.99 (default: 0.95)')
    
    args = parser.parse_args()
    
    # Validate confidence level
    if args.confidence is not None:
        if args.confidence not in [0.90, 0.95, 0.99]:
            print("Error: Confidence level must be 0.90, 0.95, or 0.99")
            return 1
    
    # Validate num_runs
    if args.num_runs is not None:
        if args.num_runs < 2:
            print("Error: Number of runs must be at least 2 for statistical analysis")
            return 1
    
    try:
        checker = PerformanceChecker(
            args.config, 
            args.verbose,
            num_runs=args.num_runs,
            confidence_level=args.confidence
        )
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
