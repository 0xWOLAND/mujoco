import jax.numpy as jp
import numpy as np
from mujoco.mjx._src.ray import ray_hfield
from mujoco.mjx._src.types import Model, Data

def make_dummy_model_and_data():
    # Create a minimal heightfield: 2x2 grid, flat at height 0.5
    m = Model()
    d = Data()
    m.hfield_size = np.array([[1.0, 1.0, 1.0, 0.0]])  # [x, y, z, unused]
    m.hfield_nrow = np.array([2])
    m.hfield_ncol = np.array([2])
    m.hfield_data = np.array([[0.5, 0.5, 0.5, 0.5]])  # flat
    m.geom_dataid = np.array([0])
    d.geom_xpos = np.array([[0.0, 0.0, 0.0]])
    d.geom_xmat = np.array([np.eye(3).flatten()])
    return m, d

def test_ray_hfield():
    m, d = make_dummy_model_and_data()
    geomid = 0
    # Ray from above, pointing down
    pnt = jp.array([0.0, 0.0, 2.0])
    vec = jp.array([0.0, 0.0, -1.0])
    dist = ray_hfield(m, d, geomid, pnt, vec)
    print("Ray-HField intersection distance:", dist)
    assert dist > 0, "Ray should intersect the heightfield"

if __name__ == "__main__":
    test_ray_hfield()