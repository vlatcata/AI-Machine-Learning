import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 # control randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # TODO: model, trainer

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y) # we use the const 20 because this is the size of each snake rectangle
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # This is a little confusing before it clicks :D
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down
        ]

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    max_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get action based on state
        action = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train the long memory, plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > max_score:
                max_score = score
                # agent.model.save()

            print('Game', agent.number_of_games, 'Score', score, 'Record', max_score)

            # TODO: plot


if __name__ == '__main__':
    train()