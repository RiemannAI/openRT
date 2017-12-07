# openRT

OpenRT is based on the [abelfunctions](https://github.com/abelfunctions/abelfunctions) package for sage where we:
- removes the sage dependency
- enable python 3 support
- perform several adjustments to speedup the RT evaluations required by [RiemannAI/theta](https://github.com/RiemannAI/theta).

# Installing

After cloning the repository compile the cython code with:

```shell
python setup.py build_ext --inplace
```

This creates the dynamic libraries that can be loaded in python by typing:

```python
import riemann_theta
```
