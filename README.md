# fastRT

An optimized version of openRT

## Installation

First step clone the theta repository:

```bash
git clone git@github.com:RiemannAI/theta.git
```

then change the URL of the git-submodule by editing .gitmodules:

```diff
diff --git a/.gitmodules b/.gitmodules
index 7230d28..df74e3a 100644
--- a/.gitmodules
+++ b/.gitmodules
@@ -1,3 +1,3 @@
 [submodule "rtbm/riemann_theta"]
        path = rtbm/riemann_theta
-       url = git@github.com:RiemannAI/openRT.git
+       url = git@github.com:RiemannAI/fastRT.git

```

Finally update and pull the git-submodule code:
```bash
git submodule sync
git submodule update --init --recursive --remote
```

Be carefull with commits, in particular with changes in the submodule and .gitmodules.
