# Cluster Analysis

A simple python package for cluster analysis in molecular dynamics trajectories.

## Usage

### Installation

```
pip install git+https://github.com/exenGT/clusteranalysis
```

### Usage

Create an example script `main.py` with the following content:

```
from cluster_analysis.cluster import analyze

analyze()
```

then run in command line prompt:

```
python main.py --help
```
to find out the available options.

### Test

To do a test, go to `cluster_analysis/data/test`, and run:

```
python test.py
```

### Details

The package separates a molecular dynamics image into individual images (\*.cif files), each one containing a separate cluster.

Meanwhile, the package returns the information about the cluster size, distribution, etc.

