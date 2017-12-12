# openRT

OpenRT provides a stand-alone implementation of the Riemann-Theta (RT) function. It is sourced from the [abelfunctions](https://github.com/abelfunctions/abelfunctions) package for sage with the following changes:

- sage dependency removed
- python 3 support added
- several improvements to speedup the RT evaluations required by [RiemannAI/theta](https://github.com/RiemannAI/theta).

# Installing

After cloning the repository compile the cython code with:

```shell
python setup.py build_ext --inplace
```

This creates the dynamic libraries that can be loaded in python by typing:

```python
import riemann_theta
```
