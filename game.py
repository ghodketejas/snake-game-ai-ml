"""
Core game logic for Snake Game AI training environment.

This module provides the SnakeGameAI class which serves as the training
environment for reinforcement learning agents. It handles game mechanics,
collision detection, reward calculation, and rendering.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time
import os

class Direction(Enum):
    """Enumeration for snake movement directions."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Color definitions
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Game constants
BLOCK_SIZE = 20
SPEED = 500
BOARD_WIDTH = 750
BOARD_HEIGHT = 750


class SnakeGameAI:
    """
    Snake game environment for AI training.
    
    Provides a complete game environment with state management, collision
    detection, reward calculation, and optional rendering for training
    reinforcement learning agents.
    """
    
    def __init__(self, delay_per_move=0):
        """
        Initialize the game environment.
        
        Args:
            delay_per_move: Additional delay between moves (seconds)
        """
        self.pygame = pygame
        self.w = BOARD_WIDTH
        self.h = BOARD_HEIGHT
        self.delay_per_move = delay_per_move
        
        # Initialize display based on environment
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            self.display = None
            self.clock = None
        else:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
            
        self.font = pygame.font.Font('arial.ttf', 25)
        self.reset()

    def reset(self):
        """
        Reset the game state to start a new episode.
        
        Initializes snake position, direction, score, and places food.
        """
        self.direction = Direction.RIGHT
        self.head = Point(
            (self.w // BLOCK_SIZE // 2) * BLOCK_SIZE,
            (self.h // BLOCK_SIZE // 2) * BLOCK_SIZE
        )
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """
        Place food at a random location not occupied by the snake.
        
        Uses recursive placement to ensure food doesn't spawn on snake.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        x = int(x)
        y = int(y)
        self.food = Point(x, y)
        
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Execute one step of the game based on the provided action.
        
        Args:
            action: Action array [straight, right, left] from the agent
            
        Returns:
            tuple: (reward, game_over, score)
        """
        self.frame_iteration += 1
        
        # Handle Pygame events
        if self.display is not None:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.pygame.quit()
                    quit()
                    
        # Control game speed
        if self.clock is not None:
            self.clock.tick(SPEED)
            
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        # Check for game over conditions
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # Check for food consumption
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            
        self._update_ui()
        
        if self.delay_per_move > 0:
            time.sleep(self.delay_per_move)
            
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Check if the given point collides with walls or snake body.
        
        Args:
            pt: Point to check (defaults to snake head)
            
        Returns:
            bool: True if collision detected
        """
        if pt is None:
            pt = self.head
            
        # Wall collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # Self collision
        if pt in self.snake[1:]:
            return True
            
        return False

    def _update_ui(self):
        """
        Update the game display with current state.
        
        Renders snake, food, and score. Does nothing in headless mode.
        """
        if self.display is None:
            return
            
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        # Draw food
        if self.food is not None:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
            
        # Draw score
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Move the snake based on the action array.
        
        Args:
            action: Action array [straight, right, left] where one element is 1
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        # Update head position
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