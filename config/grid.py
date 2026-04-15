from numpy import random

MIN_GRID_DIM = 10
MAX_GRID_DIM = 20


def random_grid_dims() -> tuple[int, int, int]:
    return (
        10,  # int(random.randint(MIN_GRID_DIM, MAX_GRID_DIM + 1)),
        20,  # int(random.randint(MIN_GRID_DIM, MAX_GRID_DIM + 1)),
        20,  # int(random.randint(MIN_GRID_DIM, MAX_GRID_DIM + 1)),
    )
