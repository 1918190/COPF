# Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation

## Environment
The code has been tested running under Python 3.6.13. The required packages are as follows:
* nvidia-tensorflow == 1.15.4+nv20.10
* tensorflow-determinism == 0.3.0
* numpy == 1.19.5
* scipy == 1.7.3

## Dataset
We provide the processed dataset: beibei.

## An example to run

* Beibei
```
python copf.py --data beibei --gnn_layer "[1, 1, 1]" --gnn_mtl_layer "[1, 1, 1]" --ssl_temp 0.1
```
