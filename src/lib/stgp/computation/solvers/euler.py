import jax
import jax.numpy as np
from jax.lax import scan

import objax

def solver_step_wrapper(m):
    def  fn(state, t):
        step_size = state['step_size']
        init_x = state['x']

        m_eval = np.array(
            m._f(init_x, t)
        )

        new_x = init_x + m_eval * step_size

        return {
            'step_size': step_size,
            'x': new_x,
        }, new_x
    return fn
    

def euler(m, init_x, num_iters, step_size):
    solver_step = solver_step_wrapper(m)
    return scan(
        solver_step,
        {'step_size': step_size, 'x': init_x},
        None,
        length=num_iters
    )
