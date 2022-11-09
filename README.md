# DANSE
Distributed Adaptive Node-specific Signal Estimation algorithm - original and added features.

As of 09.11.2022:
- Basic DANSE algorithm with fully connected network topology.
- GEVD-DANSE possible.
- Including SRO estimation and compensation as described in https://arxiv.org/abs/2211.02489

To install all dependencies --> most straightforwardly done via a Conda virtual environment:

>>> cd <your-folder-of-choice>
>>> git clone <this-repository>
>>> conda create -n <environment-name> --file req.txt
