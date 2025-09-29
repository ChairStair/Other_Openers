import numpy as np
import matplotlib.pyplot as plt

Parameters
canvas_size = (10, 10)
learning_rate = 0.1

AI "brain": starts with random weights for drawing
brain = np.random.randn(2)  # slope and intercept

Memory of experiences
memory = []

def draw_line(brain):
    x = np.arange(canvas_size[0])
    y = brain[0]x + brain[1]
    return x, y

def reward(x, y):
    # Reward for drawing near the target diagonal
    target_y = x  # perfect diagonal
    return -np.mean((y - target_y)**2)

Training loop
for step in range(1000):
    x, y = draw_line(brain)
    r = reward(x, y)
    memory.append((brain.copy(), r))
    # Learn gradually: adjust brain weights
    brain += learning_rate np.random.randn(2) * r

    if step % 200 == 0:
        plt.plot(x, y)
        plt.title(f"Step {step}")
        plt.show()

print("Final brain weights:", brain)
