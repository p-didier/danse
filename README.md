# DANSE algorithm [Paul Didier's implementation]
## Distributed Adaptive Node-specific Signal Estimation algorithm - original and added features.

As of 09.11.2022:
- Basic DANSE algorithm with fully connected network topology.
- GEVD-DANSE possible.
- Including SRO estimation and compensation as described in https://ieeexplore.ieee.org/document/10042012.

To **install** all dependencies --> most straightforwardly done via a [Conda](https://www.anaconda.com/products/distribution) virtual environment:

```
>>> cd <your-folder-of-choice>
>>> git clone <this-repository>
>>> conda create -n <environment-name> --file req.txt
```
