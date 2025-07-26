"""
Human-playable version of Snake Game using Pygame.

This module provides a standalone Snake game for human players,
separate from the AI training environment.
"""

import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

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
SPEED = 10


class SnakeGame:
    """
    Human-playable Snake game using Pygame.
    
    Provides a complete game environment for human players with
    keyboard controls, collision detection, and score tracking.
    """
    
    def __init__(self, w=1080, h=720):
        """
        Initialize the game environment.
        
        Args:
            w: Window width in pixels
            h: Window height in pixels
        """
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        
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
            
    def play_step(self):
        """
        Execute one step of the game based on user input.
        
        Returns:
            tuple: (game_over, score)
        """
        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                    
        # Move snake
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # Check for game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # Handle food consumption
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
            
        # Update display and control speed
        self._update_ui()
        self.clock.tick(SPEED)
        
        return game_over, self.score
        
    def _is_collision(self):
        """
        Check if the snake's head collides with walls or itself.
        
        Returns:
            bool: True if collision detected
        """
        # Wall collision
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # Self collision
        if self.head in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        """
        Update the game display with current state.
        
        Renders snake, food, and score.
        """
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        # Draw food
        if self.food is not None:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
            
        # Draw score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        """
        Move the snake in the specified direction.
        
        Args:
            direction: Direction enum to move the snake
        """
        x = self.head.x
        y = self.head.y
        
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)


if __name__ == '__main__':
    game = SnakeGame()
    
    # Main game loop
    while True:
        game_over, score = game.play_step()
        if game_over:
            pygame.quit()
            break
            
    print('Final Score:', score)
    pygame.quit()