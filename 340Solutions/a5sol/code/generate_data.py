# dynamics data with discontinuity due to bouncing, etc.
import numpy as np
import pickle
from tqdm import tqdm

class BallEnv:

    def __init__(self):
        self.x_min = 0  # in meters
        self.x_max = 1
        self.y_min = 0
        self.y_max = 1
        self.x = 0  # in meters
        self.y = 0
        self.dx = 0  # in meters/second
        self.dy = 0
        self.g = -9.8  # in meters/second^2
        self.time_tick = 0
        self.time_limit = 100
        self.time_scale = 0.0166

    def reset(self):
        self.x = np.random.uniform(self.x_min, self.x_max)
        self.y = np.random.uniform(self.y_min, self.y_max)
        self.dx = np.random.uniform(-1, 1)
        self.dy = np.random.uniform(-1, 1)
        self.time_tick = 0
        noise = np.random.uniform(-1, 1)
        return np.array([self.x, self.y, self.dx, self.dy, noise])

    def step(self):
        self.x += self.dx * self.time_scale
        self.y += self.dy * self.time_scale + 0.5 * self.g * self.time_scale ** 2
        self.dy += self.g * self.time_scale  # gravity
        
        # random wind
        self.dx += np.random.uniform(-0.1, 0.1)
        self.dy += np.random.uniform(-0.1, 0.1)

        # collision handling
        if self.x > self.x_max:
            self.dx *= -1
            self.x = self.x_max

        if self.x < self.x_min:
            self.dx *= -1
            self.x = self.x_min

        if self.y > self.y_max:
            self.dy *= -1
            self.y = self.y_max

        if self.y < self.y_min:
            self.dy *= -1
            self.y = self.y_min

        # random noise to throw off predictions
        noise = np.random.uniform(-1, 1)

        self.time_tick += 1
        done_yes = False
        if self.time_tick >= self.time_limit:
            done_yes = True
        return np.array([self.x, self.y, self.dx, self.dy, noise]), done_yes

def main():
    env = BallEnv()
    states = []
    next_states = []
    state = env.reset()
    for i in tqdm(range(20001)):
        states.append(state)        
        next_state, done_yes = env.step()
        next_states.append(next_state)
        if done_yes:
            state = env.reset()
        else:
            state = next_state
    states = np.stack(states)
    next_states = np.stack(next_states)

    X_train = states[:10001, :]
    y_train = next_states[:10001, 1]  # y position, tricky to guess due to gravity

    X_valid = states[10001:20001, :]
    y_valid = next_states[10001:20001, 1]  # y position, tricky to guess due to gravity


    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid
    }

    with open("../data/dynamics.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()

