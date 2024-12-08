# Spatio-Temporal Gaussian Processes (`STGP`)

`stgp` is a Gaussian process library written in `Jax`.

## Installation

### m1 specific steps

it seems that it is easier to install tensorflow properly first and then figure out jax and stgp

```bash
 conda create -n stgp_lib_only python=3.9
 conda activate stgp_lib_only
 # set up tf for m1
 conda install -c apple tensorflow-deps
 pip install tensorflow-metal
 pip install tensorflow-macos

 # pip will complain but requried for tensorflow
 pip install numpy --upgrade

 pip install tensorflow_probability

 # for m1 jax -- for some reason need to specify most recent release?
 # must use conda-forge channel for m1
 conda install jax==0.4.8 -c conda-forge

 # now that tensorflow and jax is installed properly we can install stgp
 pip install -e . 
```

## 
