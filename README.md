# Point Cloud Graph Coding
Code used for Attribute-aware Partitioning for Graph-based Point Cloud Attribute Coding.
Reuses the block GFT implementation from `https://github.com/STAC-USC/RA-GFT`.

Released under GNU General Public License version 2.

Copyright 2022 Institut für Nachrichtentechnik, RWTH Aachen University, Germany

## Install
* Create virtual environment:
```bash
    python -m venv venv
```

* Activate environment with
```bash
    source venv/bin/activate
```

* Install requirements
```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```

* Deactivate environment with
```bash
    deactivate
```
  
## Use
* Adapt the few directory locations (pointclouds, results) for your system.

* While running the virtual environment:
```bash
    python ./simulation.py --help
```

