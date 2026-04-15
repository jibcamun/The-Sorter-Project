import numpy as np
import numpy.random as random

try:
    from ..config.objects import OBJECTS, OBJECT_NAMES
    from ..config.grid import random_grid_dims
except:
    from config.objects import OBJECTS, OBJECT_NAMES
    from config.grid import random_grid_dims


def init_grid(object_names=None):
    x, y, z = random_grid_dims()

    grid = np.zeros((x, y, z))

    if object_names is None:
        n_objs = random.randint(3, len(OBJECT_NAMES) + 1)
        chosen_objs = random.choice(OBJECT_NAMES, size=n_objs, replace=False)
    else:
        chosen_objs = list(object_names)

    placed_objs = {}

    for name in chosen_objs:
        obj = OBJECTS.get(name)
        dim_x, dim_y, dim_z = obj["dims"]

        if dim_x > x or dim_y > y or dim_z > z:
            continue

        is_placed = False
        tries = 0

        while not is_placed and tries < 100:
            pos_x = random.randint(0, x - dim_x)
            pos_y = random.randint(0, y - dim_y)
            tries += 1

            candidate_zs = [0] if not obj["stack"] else range(0, z - dim_z + 1)

            for z_level in candidate_zs:
                subgrid = grid[
                    pos_x : pos_x + dim_x,
                    pos_y : pos_y + dim_y,
                    z_level : z_level + dim_z,
                ]

                if np.any(subgrid > 0):
                    continue

                if z_level > 0:
                    below = grid[
                        pos_x : pos_x + dim_x, pos_y : pos_y + dim_y, z_level - 1
                    ]
                    if np.any(below == 0):
                        continue

                grid[
                    pos_x : pos_x + dim_x,
                    pos_y : pos_y + dim_y,
                    z_level : z_level + dim_z,
                ] += 1

                placed_objs[name] = (pos_x, pos_y, z_level, obj["stack"])
                is_placed = True
                break

    return grid, placed_objs


def weighted_grid(dims: tuple):
    shape = dims if dims is not None else random_grid_dims()

    grid = random.uniform(0, 1, shape)

    x_mid, x_span, y_front = grid.shape[0] // 2, grid.shape[0] // 4, grid.shape[1] // 3
    grid[x_mid - x_span : x_mid + x_span, :y_front, :] *= 0.2

    return grid
