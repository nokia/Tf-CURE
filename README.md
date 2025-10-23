# TF-CURE

A TensorFlow-based implementation of the CURE (Clustering Using REpresentatives) algorithm for clustering n-dimensional datasets with GPU acceleration support.

## Overview

TF-CURE is a Python package that implements the CURE clustering algorithm using TensorFlow as the backend. This implementation takes advantage of GPU acceleration when available, making it suitable for large-scale clustering tasks on high-dimensional data.

The CURE algorithm is particularly effective for:
- Discovering clusters with non-spherical shapes
- Finding clusters of different sizes
- Handling outliers effectively
- Working with large datasets

## Features

- **GPU Acceleration**: Leverages TensorFlow's GPU support for faster computation
- **Flexible Configuration**: Customizable number of clusters, representatives, and compression factors
- **Compatibility**: Results are comparable with the popular `pyclustering` library
- **Comprehensive Testing**: Includes validation tests against reference implementations

## Installation

### Prerequisites

Before installing TF-CURE, ensure you have Python 3.11 or higher installed on your system.

### Using the Makefile (Recommended)

The project includes a Makefile for easy setup and installation:

1. **Install system dependencies**:
   ```bash
   make install
   ```
   This installs required system packages (python-virtualenv, python3-dev).

2. **Create and set up the virtual environment**:
   ```bash
   make virtualenv
   ```
   This creates a virtual environment in the `env/` directory and installs all required dependencies.

3. **Activate the virtual environment**:
   ```bash
   source env/bin/activate
   ```

4. **Install the package** (after activating the virtual environment):
   ```bash
   pip install -e .
   ```

## Usage

Here's a basic example of how to use TF-CURE:

```python
import numpy as np
import tensorflow as tf
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from tf_cure.src.cure import TfCURE

# Load sample data (you can use your own dataset)
input_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)

# Convert to TensorFlow tensor
with tf.device('/GPU:0'):  # Use GPU if available
    input_data_tf = tf.convert_to_tensor(input_data)

    # Create TF-CURE instance
    tf_cure_instance = TfCURE(
        data=input_data_tf,
        n_clusters=3,           # Number of clusters to find
        n_rep=5,                # Number of representatives per cluster
        compression_factor=0.5   # Compression factor for representatives
    )

    # Process the data
    tf_cure_instance.process()

    # Access results
    clusters = tf_cure_instance.clusters
    representatives = clusters.reps.numpy()

    print(f"Found {clusters.size} clusters")
    print(f"Representatives shape: {representatives.shape}")
```

### Parameters

- **data**: Input data as a TensorFlow tensor of shape `(n_samples, n_features)`
- **n_clusters**: Number of clusters to find (default: 2)
- **n_rep**: Number of representative points per cluster (default: 5)
- **compression_factor**: Factor for shrinking representatives toward cluster centroid (default: 0.5)

## Testing

The project includes tests to ensure the implementation produces results consistent with the reference `pyclustering` library.

### Running Tests

```bash
# Using the test make target to run directly with coverage configuration
make test
```

### Test Validation

The test suite validates that TF-CURE produces the same clustering results as the `pyclustering` implementation by:

1. Running both algorithms on the same dataset
2. Comparing the representative points (rounded to 4 decimal places)
3. Ensuring all representatives from the reference implementation are present in the TF-CURE results

## Project Structure

```
Tf-CURE/
├── tf_cure/                 # Main package
│   ├── __init__.py
│   ├── main.py             # Entry point
│   └── src/                # Core implementation
│       ├── cure.py         # Main CURE algorithm
│       ├── tf_cluster.py   # Cluster management
│       ├── tree.py         # R-tree implementation
│       └── contextManagers/ # Timing utilities
├── tests/                   # Test suite
│   └── test_cure_algorithm.py
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
├── Makefile               # Build and setup commands
└── README.md              # This file
```

## License

© 2025 Nokia
Licensed under the BSD 3-Clause License
SPDX-License-Identifier: BSD-3-Clause

## Contact

Mattia Milani (Nokia) - mattia.milani@nokia.com