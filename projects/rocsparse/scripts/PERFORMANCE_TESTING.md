# rocSPARSE Performance Regression Testing

A comprehensive solution for testing performance regressions in rocSPARSE with multi-architecture baseline support.

## Overview

This testing framework allows you to:
- ✅ Test a **subset of routines** (spmv, spmm, spsv, etc.) with selected matrices
- ✅ Compare against a **single baseline file** supporting multiple architectures (MI300, MI250, Navi4, etc.)
- ✅ Run tests **locally before creating PRs** or in **CI/CD pipelines**
- ✅ **Update the baseline** when performance improvements are verified
- ✅ Configurable tolerance for regression detection

## Files

- **`rocsparse-perf-check.py`** - Main testing script
- **`perf-test-config.yaml`** - Configuration file (routines, matrices, tolerance)
- **`rocsparse-perf-baseline.json`** - Single baseline file for all architectures (committed to repo)

## Quick Start

### 1. Setup

```bash
cd projects/rocsparse/scripts

# Make script executable
chmod +x rocsparse-perf-check.py

# Set environment variable for matrix directory
export ROCSPARSE_BENCH_DATA_DIR=/path/to/matrices
```

### 2. Build rocSPARSE with Benchmarks

```bash
cd projects/rocsparse

# Build with benchmarks enabled
./rmake.py --build_clients -a gfx942  # Use your GPU architecture
```

### 3. Run Regression Test

```bash
# Run performance regression test
./rocsparse-perf-check.py --config perf-test-config.yaml

# Verbose output
./rocsparse-perf-check.py --config perf-test-config.yaml -v
```

### 4. Update Baseline (After Verifying Improvements)

```bash
# Update baseline with new measurements
./rocsparse-perf-check.py --config perf-test-config.yaml --update-baseline
```

## Configuration

Edit `perf-test-config.yaml` to customize:

### Tolerance

```yaml
# Maximum allowed performance drop (percentage)
tolerance: 2.0
```

### Routines to Test

```yaml
routines:
  - name: spmv_csr
    precisions: [s, d]
  
  - name: spmm_csr
    precisions: [s, d]
    extra_args: ["--denseld", "128"]
  
  - name: spsv_csr
    precisions: [d]
```

### Matrices to Test

```yaml
matrices:
  # Direct paths
  - "$ROCSPARSE_BENCH_DATA_DIR/rma10.csr"
  - "$ROCSPARSE_BENCH_DATA_DIR/mc2depi.csr"
  
  # Wildcards
  - "$ROCSPARSE_BENCH_DATA_DIR/small/*.csr"
  - "$ROCSPARSE_BENCH_DATA_DIR/medium/*.csr"
```

## Baseline File Structure

The baseline file (`rocsparse-perf-baseline.json`) stores results for multiple architectures:

```json
{
  "version": "1.0",
  "last_updated": "2025-12-05T10:30:00",
  "architectures": {
    "mi300": {
      "spmv_csr:d:rma10": {
        "gflops": 45.2,
        "bandwidth_gb": 120.5,
        "time_ms": 0.125,
        "routine": "spmv_csr",
        "precision": "d",
        "matrix": "rma10"
      }
    },
    "mi250": {
      "spmv_csr:d:rma10": {
        "gflops": 38.5,
        "bandwidth_gb": 105.2,
        "time_ms": 0.145
      }
    },
    "navi4": {
      "spmv_csr:d:rma10": {
        "gflops": 32.1,
        "bandwidth_gb": 95.8,
        "time_ms": 0.165
      }
    }
  }
}
```

Each architecture has its own baseline, all in one file that's committed to the repository.

## Workflow

### Local Pre-Check (Before Creating PR)

```bash
#!/bin/bash
# test-before-pr.sh

# 1. Build your branch
cd projects/rocsparse
./rmake.py --build_clients -a gfx942

# 2. Run performance tests
cd scripts
./rocsparse-perf-check.py --config perf-test-config.yaml

# 3. Check exit code
if [ $? -eq 0 ]; then
  echo "✅ No regressions - safe to create PR"
else
  echo "❌ Performance regression detected - investigate before PR"
  exit 1
fi
```

### CI/CD Pipeline Integration

```yaml
# .github/workflows/performance-test.yml
name: Performance Regression Test

on: [pull_request]

jobs:
  performance-test:
    runs-on: [self-hosted, mi300]
    steps:
      - uses: actions/checkout@v3
      
      - name: Build rocSPARSE
        run: |
          cd projects/rocsparse
          ./rmake.py --build_clients -a gfx942
      
      - name: Run Performance Tests
        env:
          ROCSPARSE_BENCH_DATA_DIR: /opt/rocsparse-matrices
        run: |
          cd projects/rocsparse/scripts
          ./rocsparse-perf-check.py --config perf-test-config.yaml
      
      - name: Check Results
        if: failure()
        run: echo "Performance regression detected!"
```

### Updating Baseline After Performance Improvements

When you've verified that performance has improved:

```bash
# 1. Run tests to collect new measurements
./rocsparse-perf-check.py --config perf-test-config.yaml --update-baseline

# 2. Review the changes
git diff rocsparse-perf-baseline.json

# 3. Commit the updated baseline with your PR
git add rocsparse-perf-baseline.json
git commit -m "perf: Update baseline after spmv_csr optimization

- Improved MI300 performance by 15%
- Updated baseline measurements"
```

## Architecture Detection

The script automatically detects your GPU architecture using `rocminfo`:

- **MI300X/MI308**: Detected as `mi300`
- **MI250/MI250X**: Detected as `mi250`
- **Navi 31/32/33**: Detected as `navi31`, `navi32`, `navi33`
- **Navi 4X**: Detected as `navi4`

You can override detection:
```bash
export ROCSPARSE_TEST_ARCH=mi300
```

Or set a default in the config:
```yaml
default_architecture: "mi300"
```

## Output Examples

### Successful Test (No Regressions)

```
============================================================
Running benchmarks for architecture: mi300
============================================================

[1/6] Testing spmv_csr (precision=d) on rma10.csr
  GFlops: 45.23, Bandwidth: 120.50 GB/s, Time: 0.1250 ms

[2/6] Testing spmv_csr (precision=s) on rma10.csr
  GFlops: 52.10, Bandwidth: 135.20 GB/s, Time: 0.1100 ms

...

============================================================
Completed 6/6 benchmarks successfully
============================================================

============================================================
Comparing against baseline (tolerance: 2.0%)
============================================================

✅ spmv_csr:d:rma10 - OK (GFlops: +1.2%, Time: -1.1%)
✅ spmv_csr:s:rma10 - OK (GFlops: +0.8%, Time: -0.7%)
✅ spmm_csr:d:mc2depi - OK (GFlops: +0.3%, Time: -0.2%)

============================================================
✅ All tests PASSED - No performance regressions detected
============================================================
```

### Failed Test (Regression Detected)

```
============================================================
Comparing against baseline (tolerance: 2.0%)
============================================================

✅ spmv_csr:d:rma10 - OK (GFlops: +1.2%, Time: -1.1%)
❌ spmv_csr:s:mc2depi - REGRESSION: GFlops: -5.3%, Time: +4.8%
✅ spmm_csr:d:bmwcra_1 - OK (GFlops: +0.5%, Time: -0.4%)

============================================================
❌ Some tests FAILED - Performance regressions detected
============================================================
```

## Environment Variables

- **`ROCSPARSE_BENCH_DATA_DIR`**: Directory containing test matrices
- **`ROCSPARSE_BENCH`**: Path to `rocsparse-bench` executable (optional)
- **`ROCSPARSE_TEST_ARCH`**: Override architecture detection (optional)

## Advanced Usage

### Test Specific Routine on Specific Matrix

Edit `perf-test-config.yaml`:

```yaml
routines:
  - name: spmv_csr
    precisions: [d]

matrices:
  - "$ROCSPARSE_BENCH_DATA_DIR/scircuit.csr"
```

### Test Multiple Architectures

Run on each machine:

```bash
# On MI300 machine
./rocsparse-perf-check.py --config perf-test-config.yaml

# On MI250 machine
./rocsparse-perf-check.py --config perf-test-config.yaml

# On Navi4 machine
./rocsparse-perf-check.py --config perf-test-config.yaml
```

The same baseline file works for all architectures!

### Custom Tolerance Per Run

Override in config:

```yaml
tolerance: 1.5  # Stricter tolerance
```

## Troubleshooting

### "Could not find rocsparse-bench executable"

Set the path explicitly:

```yaml
rocsparse_bench_path: "./build/release/clients/staging/rocsparse-bench"
```

Or use environment variable:
```bash
export ROCSPARSE_BENCH=/path/to/rocsparse-bench
```

### "No matrices found matching"

Check that `ROCSPARSE_BENCH_DATA_DIR` is set:
```bash
echo $ROCSPARSE_BENCH_DATA_DIR
```

### "Architecture not found in baseline"

This architecture hasn't been tested yet. Run with `--update-baseline` to add it:

```bash
./rocsparse-perf-check.py --config perf-test-config.yaml --update-baseline
```

## Requirements

- Python 3.6+
- PyYAML: `pip install pyyaml`
- rocSPARSE built with benchmarks: `./rmake.py --build_clients`
- Test matrices in CSR or MTX format

## Contributing

When adding new routines or matrices to the test suite:

1. Update `perf-test-config.yaml`
2. Run on all supported architectures with `--update-baseline`
3. Commit both the config and updated baseline file
4. Document any new requirements in this README

## License

Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

