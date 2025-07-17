import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame modules
pygame.init()
# Set up font for displaying score
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# Enum for possible directions the snake can move
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Named tuple to represent a point on the grid
Point = namedtuple('Point', 'x, y')

# RGB color definitions
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20  # Size of each block (snake segment, food)
SPEED = 40       # Game speed (frames per second)

class SnakeGameAI:
    """
    Main class to handle the Snake game logic for AI training.
    """
    def __init__(self, w=1080, h=800):
        self.w = w
        self.h = h
        # Initialize display window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Initialize game state
        self.direction = Direction.RIGHT

        # Start snake in the center, moving right
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()  # Place the first food
        self.frame_iteration = 0  # Counts frames since last food

    def _place_food(self):
        # Randomly place food on the grid, not on the snake
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()  # Retry if food is on the snake

    def play_step(self, action):
        """
        Executes one step of the game based on the action provided by the AI.
        Returns: reward, game_over, score
        """
        self.frame_iteration += 1
        # 1. Handle user quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move the snake
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. Check if game over (collision or too many frames without eating)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10  # Negative reward for dying
            return reward, game_over, self.score

        # 4. Check if snake has eaten food
        if self.head == self.food:
            self.score += 1
            reward = 10  # Positive reward for eating food
            self._place_food()
        else:
            self.snake.pop()  # Remove last segment if not eating
        
        # 5. Update UI and control game speed
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game state
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Checks if the snake collides with the wall or itself.
        """
        if pt is None:
            pt = self.head
        # Check wall collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check self collision
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        # Draw background
        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Moves the snake in the direction based on the action.
        action: [straight, right, left] (one-hot encoded)
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Determine new direction based on action
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        # Update the head position based on direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)