# openRT

OpenRT provides a stand-alone implementation of the Riemann-Theta (RT) function. It is sourced from the [abelfunctions](https://github.com/abelfunctions/abelfunctions) package for sage with the following changes:

- sage dependency removed
- python 3 support added
- several improvements to speedup the RT evaluations required by [RiemannAI/theta](https://github.com/RiemannAI/theta).

# Installing

This package is a submodule of [RiemannAI/theta](https://github.com/RiemannAI/theta) so in order have access to the Riemann-Theta functions with python2/3 we suggest to follow the installation instructions from that project and them import:
```python
from theta.riemann_theta.riemann_theta import RiemannTheta
```

Otherwise, with python2.7 you can compile the code in place with:

```shell
python setup.py build_ext --inplace
```

This creates the dynamic libraries that can be loaded in python by typing:

```python
import riemann_theta
```
