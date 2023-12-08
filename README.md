
# About

Eikonal solvers are fun and useful.

Some uses are:
- computing signed distance functions,
- determining seismic traveltimes, and
- finding short paths for planning.

A simple way to solve the Eikonal equation is to use the Fast Sweeping Method.

See [here](https://www.ams.org/journals/mcom/2005-74-250/S0025-5718-04-01678-3/viewer/).

# Setup
```
mamba create -n peik python=3.12
```

```
mamba install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

```
mamba install -c pyviz holoviews
```