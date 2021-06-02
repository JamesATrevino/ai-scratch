# Simple pygame program

# Import and initialize the pygame library
import random

import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


board_size = 15


def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(1, 28)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='linear'))
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

    def get_vision_nw(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while x_pos > 0 and y_pos > 0:
            y_pos -= 1
            x_pos -= 1
            wall_dist += 2
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist

    def get_vision_n(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while y_pos > 0:
            y_pos -= 1
            wall_dist += 1
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist

    def get_vision_ne(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while x_pos < (board_size - 1) and y_pos > 0:
            y_pos -= 1
            x_pos += 1
            wall_dist += 2
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist

    def get_vision_e(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while x_pos < (board_size - 1):
            x_pos += 1
            wall_dist += 1
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist

    def get_vision_se(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while x_pos < (board_size - 1) and y_pos < (board_size - 1):
            y_pos += 1
            x_pos += 1
            wall_dist += 2
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist

    def get_vision_s(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while y_pos < (board_size):
            y_pos += 1
            wall_dist += 1
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist

    def get_vision_sw(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while x_pos > 0 and y_pos < (board_size - 1):
            y_pos += 1
            x_pos -= 1
            wall_dist += 2
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist

    def get_vision_w(self, board):
        x_pos, y_pos = self.get_head()['x'], self.get_head()['y']
        wall_dist = 0
        snake_dist = 0
        food_dist = 0
        while x_pos > 0:
            x_pos -= 1
            wall_dist += 1
            cell = board[x_pos][y_pos]
            if cell == "SNAKE" and snake_dist == 0:
                snake_dist = wall_dist
            elif cell == "FOOD" and food_dist == 0:
                food_dist = wall_dist

        return wall_dist, snake_dist, food_dist


def construct_board(snake, food_x, food_y):
    board = [[None for x in range(board_size + 1)] for y in range(board_size + 1)]
    for segment in snake.get_segments():
        board[segment["x"]][segment["y"]] = "SNAKE"
    board[food_x][food_y] = "FOOD"

    return board


def print_board(board):
    for y in range(board_size):
        line = list()
        for x in range(board_size):
            if board[x][y] == None:
                char = '.'
            elif board[x][y] == "SNAKE":
                char = "*"
            else:
                char = "@"
            line.append(char)
        print(" ".join(line))

    print("\n")


def get_vision(snake, food_x, food_y, debug=False):
    board = construct_board(snake, food_x, food_y)

    wall_nw, snake_nw, food_nw = snake.get_vision_nw(board)
    wall_n, snake_n, food_n = snake.get_vision_n(board)
    wall_ne, snake_ne, food_ne = snake.get_vision_ne(board)
    wall_e, snake_e, food_e = snake.get_vision_e(board)
    wall_se, snake_se, food_se = snake.get_vision_se(board)
    wall_s, snake_s, food_s = snake.get_vision_s(board)
    wall_sw, snake_sw, food_sw = snake.get_vision_sw(board)
    wall_w, snake_w, food_w = snake.get_vision_w(board)
    facing_n = 1 if snake.get_direction() == 'NORTH' else 0
    facing_e = 1 if snake.get_direction() == 'EAST' else 0
    facing_s = 1 if snake.get_direction() == 'SOUTH' else 0
    facing_w = 1 if snake.get_direction() == 'WEST' else 0

    if debug:
        print_board(board)

        s = 'wall_nw %d, snake_nw %d, food_nw %d, ' \
            '\nwall_n %d, snake_n %d, food_n %d, ' \
            '\nwall_ne %d, snake_ne %d, food_ne %d, ' \
            '\nwall_e %d, snake_e %d, food_e %d, ' \
            '\nwall_se %d, snake_se %d, food_se %d, ' \
            '\nwall_s %d, snake_s %d, food_s %d, ' \
            '\nwall_sw %d, snake_sw %d, food_sw %d, ' \
            '\nwall_w %d, snake_w %d, food_w %d, ' \
            '\nfacing_n %d, facing_e %d, facing_s %d, facing_w %d' % \
            (wall_nw, snake_nw, food_nw,
             wall_n, snake_n, food_n,
             wall_ne, snake_ne, food_ne,
             wall_e, snake_e, food_e,
             wall_se, snake_se, food_se,
             wall_s, snake_s, food_s,
             wall_sw, snake_sw, food_sw,
             wall_w, snake_w, food_w,
             facing_n, facing_e, facing_s, facing_w)
        print(s)

    return np.array([wall_nw, snake_nw, food_nw,
                     wall_n, snake_n, food_n,
                     wall_ne, snake_ne, food_ne,
                     wall_e, snake_e, food_e,
                     wall_se, snake_se, food_se,
                     wall_s, snake_s, food_s,
                     wall_sw, snake_sw, food_sw,
                     wall_w, snake_w, food_w,
                     facing_n, facing_e, facing_s, facing_w])


def new_food(board_size, snake):
    import itertools
    import random
    available_coordinates = list(itertools.product(range(board_size), range(board_size)))
    for segment in snake.segments:
        if (segment['x'], segment['y']) in available_coordinates:
            available_coordinates.remove((segment['x'], segment['y']))

    return random.choice(available_coordinates)


def run():

    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
            epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = Adam(learning_rate=0.00025, clipnorm=1.0)

    batch_size=32
    model = build_model()
    model_target = build_model()

    pygame.init()
    # Set up the drawing window
    screen_width = screen_height = 500
    screen_buffer = 50
    game_width = screen_width - (2 * screen_buffer)
    game_height = screen_height - (2 * screen_buffer)

    cell_size = game_width / board_size

    color_black = (5, 5, 5)
    color_blue = (0, 0, 255)
    screen = pygame.display.set_mode([screen_width, screen_height])
    clock = pygame.time.Clock()

    frameRate = 60

    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    epsilon_random_frames = 100
    epsilon_greedy_frames = 9000

    max_memory_length = 10000000
    update_after_actions = 100
    update_target_network = 1000
    loss_function = MeanSquaredError()
    max_steps_per_episode = 10000
    num_actions=4

    while True:
        snake = Snake(int(board_size / 2), int(board_size / 2), 2)
        food_x, food_y = new_food(board_size, snake)
        episode_reward = 0
        game_over = False
        state = get_vision(snake, food_x, food_y)
        timestep = 1

        while timestep < max_steps_per_episode:
            clock.tick(frameRate)
            frame_count += 1
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # apply sampled action
            if action == 0:
                snake.move_north()
            elif action == 1:
                snake.move_east()
            elif action == 2:
                snake.move_south()
            elif action == 3:
                snake.move_west()

            snake.move()
            head_x = snake.get_head()['x']
            head_y = snake.get_head()['y']
            if (head_x == board_size) or (head_x == -1):
                game_over = True
                reward = -1
            elif (head_y == board_size) or (head_y == -1):
                game_over = True
                reward = -1
            elif head_x == food_x and head_y == food_y:
                snake.grow()
                food_x, food_y = new_food(board_size, snake)
                reward = 10
            # timestep = 1
            else:
                reward = 0
            # Snake eats itself
            for segment in snake.get_segments()[1:]:
                if head_x == segment['x'] and head_y == segment['y']:
                    game_over = True
                    reward = -1

            state_next = get_vision(snake, food_x, food_y)
            episode_reward += reward
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(game_over)
            rewards_history.append(reward)
            state = state_next
            timestep += 1

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)
                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if game_over:
                break

            # Fill the background with white
            screen.fill((255, 255, 255))
            font = pygame.font.Font('freesansbold.ttf', 32)
            text = font.render('Score: %d ' % (snake.get_length() - snake.get_initial_length()), True, color_black)
            textRect = text.get_rect()
            textRect.center = (100, 15)
            screen.blit(text, textRect)

            # Allow user to quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and not game_over:
                        snake.turn_left()
                    elif event.key == pygame.K_RIGHT and not game_over:
                        snake.turn_right()

            # draw rect to box in game
            pygame.draw.rect(screen, color_black, (screen_buffer, screen_buffer, game_width, game_height), width=2)
            for i in range(board_size):
                # horizontal line
                pygame.draw.line(screen, color_black, (screen_buffer, i * cell_size + screen_buffer),
                                 (screen_width - screen_buffer, i * cell_size + screen_buffer))
                # vertical line
                pygame.draw.line(screen, color_black, (i * cell_size + screen_buffer, screen_buffer),
                                 (i * cell_size + screen_buffer, screen_height - screen_buffer))

                # Draw snake
            for segment in snake.get_segments():
                pygame.draw.circle(screen, color_blue, ((segment['x'] * cell_size) + screen_buffer + (cell_size / 2),
                                                        (segment['y'] * cell_size) + screen_buffer + (cell_size / 2)),
                                   cell_size / 2)

                # Draw food
            pygame.draw.circle(screen, color_black, (
                (food_x * cell_size) + screen_buffer + (cell_size / 2), (food_y * cell_size) + screen_buffer + (cell_size / 2)),
                               cell_size / 2)

            # Flip the display
            pygame.display.flip()

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 1000:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        episode_count += 1
        if running_reward > 100:
            print("solved at episode {}!".format(episode_count))
            break
    # Done! Time to quit.
    pygame.quit()


if __name__ == "__main__":
    run()
