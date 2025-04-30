import typer
import numpy as np
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import stgp
from stgp.transforms.pdes import  DampedPendulum1D
from stgp.computation.solvers.euler import euler



def main(g: float = None, l: float = None, dt: float = None, b: float = None, n: int = None, plot: bool = False):
    root = Path('data')
    root.mkdir(exist_ok=True)

    g = float(g)
    l = float(l)
    dt = float(dt)
    n = int(n)
    b = float(b)

    theta_init=np.pi*3/4
    theta_d_init=0

    pde = DampedPendulum1D(None, g=g, l=l, b=b, train=False)
    pde.print()
    res, init_x = euler(pde, np.array([theta_init, theta_d_init]), n, dt)

    Y = init_x[:, 0][:, None]
    X = (np.arange(n)*dt)[:, None]

    data = np.hstack([X, Y])
    data_df = pd.DataFrame(data, columns=['x', 'y'])

    if plot:
        plt.plot(data_df['x'], data_df['y'])
        plt.show()

    name = root / f'pendulum_dt_{dt}_g_{g}_l_{l}_b_{b}_n_{n}.csv'
    print('saving to: ', name)
    data_df.to_csv(name, index=False)


if __name__ == "__main__":
    typer.run(main)
