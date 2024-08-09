import numpy as np
import pygame
import sys

class MinecraftCartEnv:

# ------ Initialise game, load images and set starting state ------

    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Set up the display
        self.mapWidth, self.mapHeight = 340, 460
        self.windWidth, self.windHeight = self.mapWidth, self.mapHeight
        self.screen = pygame.display.set_mode((self.mapWidth, self.mapHeight))
        pygame.display.set_caption("Map Navigation Game")

        # Load and scale the map
        self.map = pygame.transform.scale(pygame.image.load("map_340_460.png"), (self.mapWidth, self.mapHeight))

        # Load and scale the images
        self.agentImg = pygame.transform.scale(pygame.image.load("agent.png"), (20, 20))
        self.diamondBlockImg = pygame.transform.scale(pygame.image.load("diamondBlock.png"), (20, 20))
        self.trackImg = pygame.transform.scale(pygame.image.load("track.png"), (20, 20))
        self.distractorsImg = pygame.transform.scale(pygame.image.load("distractor.png"), (20, 20))
        self.floorImg = pygame.transform.scale(pygame.image.load("floor.png"), (20, 20))
        self.pickaxeImg = pygame.transform.scale(pygame.image.load("pickaxe.png"), (20, 20))
        self.cartImg = pygame.transform.scale(pygame.image.load("cart.png"), (20, 20))
        self.spadeImg = pygame.transform.scale(pygame.image.load("spade.png"), (20, 20))

        # Remove black backgrounds
        self.agentImg.set_colorkey((0, 0, 0))
        self.pickaxeImg.set_colorkey((0, 0, 0))
        self.spadeImg.set_colorkey((0, 0, 0))

        # Find starting position (pink tile)
        self.startPos = self.find_start()

        # Set up the agent
        self.agentRect = self.agentImg.get_rect()

        # Define colours
        self.initialWalkableColours = [(199, 133, 60), (40, 126, 117), (255, 192, 203), (0, 209, 255), (202, 202, 202), (255, 245, 0), (79, 199, 60), (149, 143, 143), (66, 0, 255)]
        self.floorColour = [(199, 133, 60), (255, 192, 203), (0, 209, 255)]
        self.specialColour = (40, 126, 117)
        self.diamondColour = (0, 209, 255)
        self.trackColour = (202, 202, 202)
        self.distractorColour = (255, 0, 0)
        self.pickaxeColour = (255, 245, 0)
        self.cartColour = (149, 143, 143)
        self.spadeColour = (79, 199, 60)
        self.trackStickColour = (66, 0, 255)
        self.walkableColours = self.initialWalkableColours.copy()

        # Initialize positions
        self.initialise_positions()

        self.maxSteps = 40 # 40 step rollout as in paper

        self.clock = pygame.time.Clock()
        self.set_game()

# ------ Game update triggered by action with optional rendering ------

    def step(self, move, strafe, use, render=True):

        # all actions handled in discrete movements not continuous
        
        x, y = self.agentRect.topleft

        if strafe == -1:
            x = x - 20
            newPos = (x, y)
        elif strafe == 1:
            x = x + 20
            newPos = (x, y)

        if move == 1:
            y = y - 20
            newPos = (x, y)
        elif move == -1:
            y = y + 20
            newPos = (x, y)

        if use == 1:
            self.use_action()
            newPos = (x, y)

        if self.is_walkable(newPos):
            if newPos == self.cartPos:
                if strafe == -1 and self.move_cart(-1):
                    self.agentRect.topleft = newPos
                elif strafe == 1 and self.move_cart(1):
                    self.agentRect.topleft = newPos
            else:
                self.agentRect.topleft = newPos

            self.water_trap(self.map.get_at(newPos)[:3])
            self.check_pickaxe_collection(newPos)
            self.check_spade_collection(newPos)

        self.update_cart_pos()

        observation = self.observation()
        done = self.stepCount + 1 >= self.maxSteps

        if render:
            self.render()

        self.stepCount += 1
        
        if done:
            self.set_game()  # Reset the game state for the next episode

        return observation, done
    
# ------ Game Utility Functions ------

    # Utilities for returning observations from environment 

    def observation(self):
        state = [
            *self.normalize_position(self.agentRect.topleft),
            *self.normalize_position(self.pickaxePos),
            *self.normalize_position(self.spadePos),
            self.normalize_cart_position(self.cartPos),
            *[coord for pos in self.distractorPos for coord in self.normalize_position(pos)],
            *[1 if pos in self.diamondPos else 0 for pos in self.diamondPos_all]
        ]
        return np.array(state, dtype=np.float32)

    # all values normalised between 0 and 1 and rounded to 3sf

    def normalize_cart_position(self, pos):
        return round((pos[0] / self.mapWidth), 3)

    def normalize_position(self, pos):
        if pos is self.pickaxePos and self.pickaxePos is None:
            pos = self.agentRect.topleft
        elif pos is self.spadePos and self.spadePos is None:
            pos = self.agentRect.topleft
        return round((pos[0] / self.mapWidth), 3), round((pos[1] / self.mapHeight), 3)

    # Utilities used in finding and initialising positions of game elements

    def find_start(self):
        for y in range(0, self.mapHeight, 20):
            for x in range(0, self.mapWidth, 20):
                if self.map.get_at((x, y)) == (255, 192, 203):
                    return (x, y)
        raise ValueError("Error: Could not find starting position")

    def initialise_positions(self):
        self.diamondPos = set()
        self.diamondPos_all = set()
        self.trackPos = set()
        self.distractorPos = set()
        self.floorPos = set()
        self.pickaxePos = None
        self.cartPos = None
        self.spadePos = None
        self.specialPos = set()

        for y in range(0, self.mapHeight, 20):
            for x in range(0, self.mapWidth, 20):
                colour = self.map.get_at((x, y))[:3]
                if colour == self.diamondColour:
                    self.diamondPos.add((x, y))
                    self.diamondPos_all.add((x,y))
                elif colour == self.trackColour or colour == self.cartColour or colour == self.trackStickColour:
                    self.trackPos.add((x, y))
                elif colour == self.distractorColour:
                    self.distractorPos.add((x, y))
                elif colour in self.floorColour or colour == self.pickaxeColour or colour == self.spadeColour:
                    self.floorPos.add((x, y))
                elif colour == self.specialColour:
                    self.specialPos.add((x, y))
                if colour == self.pickaxeColour:
                    self.pickaxePos = (x, y)
                    self.initialPickaxePos = (x, y)
                elif colour == self.cartColour:
                    self.cartPos = (x, y)
                    self.initialCartPos = (x, y)
                elif colour == self.spadeColour:
                    self.spadePos = (x, y)
                    self.initialSpadePos = (x, y)

    # Utility for setting initial game state

    def set_game(self):
        self.agentRect.topleft = self.startPos
        self.hasPickaxe = False
        self.hasSpade = False
        self.cartPos = self.initialCartPos
        self.cartStuck = False
        self.pickaxePos = self.initialPickaxePos
        self.spadePos = self.initialSpadePos
        self.walkableColours = self.initialWalkableColours.copy()
        self.stepCount = 0

    # Utility for removing diamond block if on adjacent square

    def use_action(self):
        if self.hasPickaxe:
            for dx, dy in [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20)]:
                pos = (self.agentRect.left + dx, self.agentRect.top + dy)
                if pos in self.diamondPos:
                    self.diamondPos.remove(pos)
    
    # Utility for optional game rendering

    def render(self):
        self.screen.blit(self.map, (0, 0))
        for pos in self.floorPos:
            self.screen.blit(self.floorImg, pos)
        for pos in self.trackPos:
            self.screen.blit(self.trackImg, pos)
        for pos in self.distractorPos:
            self.screen.blit(self.distractorsImg, pos)
        for pos in self.diamondPos:
            self.screen.blit(self.diamondBlockImg, pos)
        if self.pickaxePos:
            self.screen.blit(self.pickaxeImg, self.pickaxePos)
        if self.spadePos:
            self.screen.blit(self.spadeImg, self.spadePos)
        if self.cartPos:
            self.screen.blit(self.cartImg, self.cartPos)
        self.screen.blit(self.agentImg, self.agentRect)
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

    # Utilities for defining movement

    def is_walkable(self, pos):
        if pos in self.diamondPos:
            return False
        try:
            colour = self.map.get_at(pos)
            return colour[:3] in self.walkableColours
        except IndexError:
            return False

    def water_trap(self, currentTileColour):
        if currentTileColour == self.specialColour:
            self.walkableColours = [self.specialColour]
        else:
            self.walkableColours = self.initialWalkableColours.copy()

    def move_cart(self, direction):
        if self.cartStuck:
            return False
        newCartPos = (self.cartPos[0] + direction * 20, self.cartPos[1])
        if newCartPos in self.trackPos:
            self.cartPos = newCartPos
            if self.cartPos in self.specialPos:
                self.cartStuck = True
            return True
        return False

    def update_cart_pos(self):
        if self.cartStuck:
            return

        agentAdjacent = any((self.agentRect.left + dx, self.agentRect.top + dy) == self.cartPos 
                             for dx, dy in [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20)])

        if not agentAdjacent:
            dx = self.initialCartPos[0] - self.cartPos[0]
            if dx != 0:
                dx = dx // abs(dx) * 20
                newCartPos = (self.cartPos[0] + dx, self.cartPos[1])
                if newCartPos in self.trackPos:
                    self.cartPos = newCartPos
                    if self.cartPos in self.specialPos:
                        self.cartStuck = True

    # utility for collecting pickaxe or spade

    def check_pickaxe_collection(self, newPos):
        if newPos == self.pickaxePos:
            if self.hasSpade:
                self.spadePos = self.initialSpadePos
                self.hasSpade = False
            self.pickaxePos = None
            self.hasPickaxe = True

    def check_spade_collection(self, newPos):
        if newPos == self.spadePos:
            if self.hasPickaxe:
                self.pickaxePos = self.initialPickaxePos
                self.hasPickaxe = False
            self.spadePos = None
            self.hasSpade = True

    def close(self):
        pygame.quit()
        sys.exit()