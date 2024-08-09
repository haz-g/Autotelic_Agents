import numpy as np

def action_to_string(action):
    if np.array_equal(action, [-1, 0, 0]):
        return "forward"
    elif np.array_equal(action, [1, 0, 0]):
        return "backward"
    elif np.array_equal(action, [0, -1, 0]):
        return "left"
    elif np.array_equal(action, [0, 1, 0]):
        return "right"
    elif np.array_equal(action, [0, 0, 1]):
        return "use"
    else:
        return "unknown"