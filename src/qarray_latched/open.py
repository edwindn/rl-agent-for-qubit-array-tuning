"""
Author: b-vanstraaten
Date: 31/03/2025
"""

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxopt import BoxOSQP
import jax

import matplotlib.pyplot as plt

from functions import *



class LatchedDoubleDot:

    def __init__(self, cdd, cgd, tc):
        self.cdd = cdd
        self.cdd_inv = jnp.linalg.inv(cdd)
        self.cgd = cgd
        self.tc = tc

    def ground_state_nd(self, vg):
        return ground_state_nd(vg, self.cgd, self.cdd_inv, self.tc)

    def latched_csd(self, vg):
        """
        Computes the continuous charge distribution for a latched double dot.
        :param vg: the dot voltage coordinate vector
        :return: the continuous charge distribution
        """
        return latched_2d(vg, self.cgd, self.cdd_inv, self.tc)


cdd = jnp.array(
    [
        [1, -0.1],
        [-0.1, 1]
    ]
)

cgd = jnp.array(
    [
        [1, 0.1],
        [0.1, 1]
    ]
)

vg = jnp.stack(
    jnp.meshgrid(
        np.linspace(-1, 2, 30),
        np.linspace(-1, 2, 30),
    ),
    axis=-1
)

model = LatchedDoubleDot(cdd, cgd, tc=0.1)

n = model.latched_csd(vg)


plt.imshow(n[:, :, 0], extent=(-1, 1, -1, 1), origin='lower')
plt.colorbar()
plt.title('Charge distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()