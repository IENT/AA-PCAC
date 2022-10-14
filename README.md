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

* Activate the virtual environment with
```bash
source venv/bin/activate
```

* Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

* When done, deactivate the environment with
```bash
deactivate
```
  
## Use
* Adapt the directory locations for input pointclouds and results for your system.

* While the virtual environment is active:
```bash
python ./simulation.py --help
```

