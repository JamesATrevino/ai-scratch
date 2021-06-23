import gym
import pygame
import numpy as np
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.core import Env


def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(1, 11)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='linear'))
    return model


class Snake(object):
    def __init__(self, x_pos, y_pos, initial_length):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.initial_length = initial_length
        self.direction = 'EAST'
        self.segments = list()
        for i in range(initial_length):
            self.segments.append({"x": x_pos, "y": y_pos})

    def get_direction(self):
        return self.direction

    def get_segments(self):
        return self.segments

    def get_length(self):
        return len(self.segments)

    def get_initial_length(self):
        return self.initial_length

    def get_head(self):
        return self.segments[0]

    def turn_left(self):
        if self.direction == 'SOUTH':
            self.direction = 'EAST'
        elif self.direction == 'EAST':
            self.direction = 'NORTH'
        elif self.direction == 'NORTH':
            self.direction = 'WEST'
        elif self.direction == 'WEST':
            self.direction = 'SOUTH'

    def turn_right(self):
        if self.direction == 'SOUTH':
            self.direction = 'WEST'
        elif self.direction == 'WEST':
            self.direction = 'NORTH'
        elif self.direction == 'NORTH':
            self.direction = 'EAST'
        elif self.direction == 'EAST':
            self.direction = 'SOUTH'

    def move_north(self):
        if not self.direction == 'SOUTH':
            self.direction = 'NORTH'

    def move_east(self):
        if not self.direction == 'WEST':
            self.direction = 'EAST'

    def move_south(self):
        if not self.direction == 'NORTH':
            self.direction = 'SOUTH'

    def move_west(self):
        if not self.direction == 'EAST':
            self.direction = 'WEST'

    def move(self):
        import copy
        tmp = copy.deepcopy(self.segments)
        # move children
        for i in range(1, len(self.segments)):
            parent = tmp[i - 1]
            self.segments[i]["x"] = parent["x"]
            self.segments[i]["y"] = parent["y"]

        # move head
        if self.direction == 'SOUTH':
            self.segments[0]["y"] += 1
        elif self.direction == 'EAST':
            self.segments[0]["x"] += 1
        elif self.direction == 'NORTH':
            self.segments[0]["y"] -= 1
        elif self.direction == 'WEST':
            self.segments[0]["x"] -= 1

    def grow(self):
        self.segments.append({"x": self.segments[-1]["x"], "y": self.segments[-1]["y"]})


class SnakeEnv(Env):
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.viewer = None
        self.snake = Snake(int(board_size / 2), int(board_size / 2), 2)
        self.food_x, self.food_y = self.new_food()
        self.state = self.get_state()
        self.steps_left = 300
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # apply sampled action
        if action == 0:
            self.snake.turn_left()
        elif action == 1:
            self.snake.turn_right()

        self.snake.move()
        head_x = self.snake.get_head()['x']
        head_y = self.snake.get_head()['y']
        # out of bounds
        if (head_x == self.board_size) or (head_x == -1):
            done = True
            reward = -10
        elif (head_y == self.board_size) or (head_y == -1):
            done = True
            reward = -10
        # found food
        elif head_x == self.food_x and head_y == self.food_y:
            self.snake.grow()
            self.food_x, self.food_y = self.new_food()
            reward = 10
            done = False
            self.steps_left = 300
        else:
            reward = 0
            done = False
            self.steps_left -= 1
        # Snake eats itself
        for segment in self.snake.get_segments()[1:]:
            if head_x == segment['x'] and head_y == segment['y']:
                done = True
                reward = -10

        if self.steps_left == 0:
            done = True

        self.state = self.get_state()
        return self.state, reward, done, {'score': self.snake.get_length() - self.snake.initial_length}

    def reset(self):
        self.snake = Snake(int(self.board_size / 2), int(self.board_size / 2), 2)
        self.food_x, self.food_y = self.new_food()
        self.state = self.get_state()
        self.steps_left = 300
        return self.state

    def render(self, mode='human', close=False):
        if self.viewer is None:
            pygame.init()
            self.viewer = 'running'

        frameRate = 60
        # Set up the drawing window
        clock = pygame.time.Clock()
        clock.tick(frameRate)
        screen_width = screen_height = 500
        screen_buffer = 50
        game_width = screen_width - (2 * screen_buffer)
        game_height = screen_height - (2 * screen_buffer)

        cell_size = game_width / self.board_size

        color_black = (5, 5, 5)
        color_blue = (0, 0, 255)
        screen = pygame.display.set_mode([screen_width, screen_height])
        # Fill the background with white
        screen.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Score: %d ' % (self.snake.get_length() - self.snake.get_initial_length()), True, color_black)
        textRect = text.get_rect()
        textRect.center = (100, 15)
        screen.blit(text, textRect)

        # draw rect to box in game
        pygame.draw.rect(screen, color_black, (screen_buffer, screen_buffer, game_width, game_height), width=2)
        for i in range(self.board_size):
            # horizontal line
            pygame.draw.line(screen, color_black, (screen_buffer, i * cell_size + screen_buffer),
                             (screen_width - screen_buffer, i * cell_size + screen_buffer))
            # vertical line
            pygame.draw.line(screen, color_black, (i * cell_size + screen_buffer, screen_buffer),
                             (i * cell_size + screen_buffer, screen_height - screen_buffer))

            # Draw snake
        for segment in self.snake.get_segments():
            pygame.draw.circle(screen, color_blue, ((segment['x'] * cell_size) + screen_buffer + (cell_size / 2),
                                                    (segment['y'] * cell_size) + screen_buffer + (cell_size / 2)),
                               cell_size / 2)

            # Draw food
        pygame.draw.circle(screen, color_black, (
            (self.food_x * cell_size) + screen_buffer + (cell_size / 2), (self.food_y * cell_size) + screen_buffer + (cell_size / 2)),
                           cell_size / 2)

        # Flip the display
        pygame.display.flip()

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None

    def construct_board(self):
        board = [[None for x in range(self.board_size + 1)] for y in range(self.board_size + 1)]
        for segment in self.snake.get_segments():
            board[segment["x"]][segment["y"]] = "SNAKE"
        board[self.food_x][self.food_y] = "FOOD"

        return board

    def print_board(self, board):
        for y in range(self.board_size):
            line = list()
            for x in range(self.board_size):
                if board[x][y] is None:
                    char = '.'
                elif board[x][y] == "SNAKE":
                    char = "*"
                else:
                    char = "@"
                line.append(char)
            print(" ".join(line))

        print("\n")

    def get_state(self):
        board = self.construct_board()
        direction = self.snake.get_direction()
        head_x = self.snake.get_head()['x']
        head_y = self.snake.get_head()['y']

        if direction == 'NORTH':
            danger_ahead = head_y == 0 or board[head_x][head_y - 1] == 'SNAKE'
            danger_left = head_x == 0 or board[head_x - 1][head_y] == 'SNAKE'
            danger_right = head_x == self.board_size or board[head_x + 1][head_y] == 'SNAKE'
        elif direction == 'EAST':
            danger_ahead = head_x == self.board_size or board[head_x + 1][head_y] == 'SNAKE'
            danger_left = head_y == 0 or board[head_x][head_y - 1] == 'SNAKE'
            danger_right = head_y == self.board_size or board[head_x][head_y + 1] == 'SNAKE'
        elif direction == 'SOUTH':
            danger_ahead = head_y == self.board_size or board[head_x][head_y + 1] == 'SNAKE'
            danger_left = head_x == 0 or board[head_x - 1][head_y] == 'SNAKE'
            danger_right = head_x == self.board_size or board[head_x + 1][head_y] == 'SNAKE'
        else:
            danger_ahead = head_x == 0 or board[head_x - 1][head_y] == 'SNAKE'
            danger_left = head_y == self.board_size or board[head_x][head_y + 1] == 'SNAKE'
            danger_right = head_y == 0 or board[head_x][head_y - 1] == 'SNAKE'

        facing_n = direction == 'NORTH'
        facing_e = direction == 'EAST'
        facing_s = direction == 'SOUTH'
        facing_w = direction == 'WEST'
        food_west = self.food_x < self.snake.get_head()['x']
        food_north = self.food_y < self.snake.get_head()['y']
        food_east = self.food_x > self.snake.get_head()['x']
        food_south = self.food_y > self.snake.get_head()['y']

        return np.array([danger_ahead, danger_left, danger_right, facing_n, facing_e, facing_s, facing_w, food_west, food_north, food_east, food_south], dtype=int)

    def new_food(self):
        import itertools
        import random
        available_coordinates = list(itertools.product(range(self.board_size), range(self.board_size)))
        for segment in self.snake.segments:
            if (segment['x'], segment['y']) in available_coordinates:
                available_coordinates.remove((segment['x'], segment['y']))

        return random.choice(available_coordinates)


def run():
    env = SnakeEnv(15)
    model = build_model()
    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, policy=policy, nb_actions=env.action_space.n)
    sarsa.compile(Adam(), metrics=['mse'])
    sarsa.fit(env, nb_steps=200000, visualize=False, verbose=1)
    sarsa.save_weights('sarsa_snake_naive_3_moves.h5f', overwrite=True)
    scores = sarsa.test(env, nb_episodes=100, visualize=True, verbose=1)
    print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))


if __name__ == "__main__":
    run()
