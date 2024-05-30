import pygame
from pygame import mixer


class Pacman:
    def __init__(
        self, sprite: pygame.Surface, spawn: tuple[int, int], direction: int = 0
    ) -> None:
        self.sprites = [
            pygame.transform.rotate(sprite, 90),
            pygame.transform.rotate(sprite, -90),
            pygame.transform.rotate(sprite, 180),
            pygame.transform.rotate(sprite, 0),
        ]
        self.x = spawn[0]
        self.y = spawn[1]
        self.direction = direction
        self.aggressiveness: int = 0

    def location(self):
        return self.x, self.y

    def go_up(self, grid):
        self.direction = 0
        if grid[self.y - 1][self.x] != 1:
            self.y -= 1

    def go_down(self, grid):
        self.direction = 1
        if grid[self.y + 1][self.x] != 1:
            self.y += 1

    def go_left(self, grid):
        self.direction = 2
        if grid[self.y][self.x - 1] != 1:
            self.x -= 1

    def go_right(self, grid):
        self.direction = 3
        if grid[self.y][self.x + 1] != 1:
            self.x += 1

    def move(self, grid):
        # up: 0, down: 1, left: 2, right: 3
        if self.direction == 0:
            self.go_up(grid)
        elif self.direction == 1:
            self.go_down(grid)
        elif self.direction == 2:
            self.go_left(grid)
        elif self.direction == 3:
            self.go_right(grid)

    def aggressive(self):
        return self.aggressiveness > 0


class Ghost:
    def __init__(
        self,
        normal_sprite: pygame.Surface,
        scared_sprite: pygame.Surface,
        spawn: tuple[int, int],
    ):
        self.normal_sprite = normal_sprite
        self.scared_sprite = scared_sprite
        self.spawn = spawn
        self.x = spawn[0]
        self.y = spawn[1]

    def location(self):
        return self.x, self.y

    def go_up(self, grid):
        if grid[self.y - 1][self.x] != 1:
            self.y -= 1

    def go_down(self, grid):
        if grid[self.y + 1][self.x] != 1:
            self.y += 1

    def go_left(self, grid):
        if grid[self.y][self.x - 1] != 1:
            self.x -= 1

    def go_right(self, grid):
        if grid[self.y][self.x + 1] != 1:
            self.x += 1

    def copy(self):
        return Ghost(
            self.normal_sprite, self.scared_sprite, (self.spawn[0], self.spawn[1])
        )


# 0: empty, 1: wall, 2: small_dot, 3: big_dot
GRID = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 0, 1],
    [1, 0, 1, 1, 1, 2, 1, 1, 2, 1],
    [1, 2, 1, 2, 2, 2, 2, 1, 2, 1],
    [1, 2, 2, 2, 1, 1, 2, 1, 2, 1],
    [1, 2, 1, 2, 2, 2, 2, 1, 2, 1],
    [1, 2, 1, 1, 1, 1, 0, 2, 2, 1],
    [1, 0, 2, 2, 2, 1, 2, 1, 2, 1],
    [1, 1, 2, 1, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

COLOR_MAP = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 255, 255),
    3: (255, 200, 200),
    "pacman": (255, 255, 0),
}
SPAWN = (4, 3)
TILE_SIZE = 50
SMALL_DOT_RADIUS = 5
BIG_DOT_RADIUS = 15
AGGRESSIVENESS_COOL_DOWN = 12
FPS = 2.5
WIDTH = len(GRID[0])
HEIGHT = len(GRID)


def draw_grid(screen: pygame.Surface, pacman: Pacman, ghosts: list[Ghost]):
    for y, row in enumerate(GRID):
        for x, tile in enumerate(row):
            color = COLOR_MAP[tile]
            if tile == 0 or tile == 1:
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )
            elif tile == 2:
                pygame.draw.rect(
                    screen,
                    COLOR_MAP[0],
                    pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )
                pygame.draw.circle(
                    screen,
                    color,
                    (x * TILE_SIZE + TILE_SIZE / 2, y * TILE_SIZE + TILE_SIZE / 2),
                    SMALL_DOT_RADIUS,
                )
            elif tile == 3:
                pygame.draw.rect(
                    screen,
                    COLOR_MAP[0],
                    pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                )
                pygame.draw.circle(
                    screen,
                    color,
                    (x * TILE_SIZE + TILE_SIZE / 2, y * TILE_SIZE + TILE_SIZE / 2),
                    BIG_DOT_RADIUS,
                )

            for ghost in ghosts:
                if pacman.aggressive():
                    screen.blit(
                        ghost.scared_sprite, (ghost.x * TILE_SIZE, ghost.y * TILE_SIZE)
                    )
                else:
                    screen.blit(
                        ghost.normal_sprite, (ghost.x * TILE_SIZE, ghost.y * TILE_SIZE)
                    )
            screen.blit(
                pacman.sprites[pacman.direction],
                (pacman.x * TILE_SIZE, pacman.y * TILE_SIZE),
            )


def draw_score(screen: pygame.Surface, font: pygame.font.Font, score: int):
    text_surface = font.render(f"Score: {score}", False, (255, 255, 255))
    screen.blit(text_surface, (0, 0))


pygame.init()
mixer.init()
pygame.font.init()

clock = pygame.time.Clock()
screen = pygame.display.set_mode([WIDTH * TILE_SIZE, HEIGHT * TILE_SIZE])

pacman_sprite = pygame.transform.scale(
    pygame.image.load(r".\assets\pacman.png").convert(), (TILE_SIZE, TILE_SIZE)
)
blinky_sprite = pygame.transform.scale(
    pygame.image.load(r".\assets\blinky.png").convert(), (TILE_SIZE, TILE_SIZE)
)
pinky_sprite = pygame.transform.scale(
    pygame.image.load(r".\assets\pinky.png").convert(), (TILE_SIZE, TILE_SIZE)
)
inky_sprite = pygame.transform.scale(
    pygame.image.load(r".\assets\inky.png").convert(), (TILE_SIZE, TILE_SIZE)
)
clyde_sprite = pygame.transform.scale(
    pygame.image.load(r".\assets\clyde.png").convert(), (TILE_SIZE, TILE_SIZE)
)
scared_ghost_sprite = pygame.transform.scale(
    pygame.image.load(r".\assets\scared_ghost.png").convert(), (TILE_SIZE, TILE_SIZE)
)

mixer.music.load(r".\assets\pacman_music.mp3")
mixer.music.play()

font = pygame.font.SysFont("Comic Sans MS", 30)

pacman = Pacman(pacman_sprite, SPAWN, 3)
blinky = Ghost(blinky_sprite, scared_ghost_sprite, (1, 1))
ghosts = [blinky]
score = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # up: 0, down: 1, left: 2, right: 3
            if event.key == pygame.K_w:
                pacman.direction = 0
            elif event.key == pygame.K_s:
                pacman.direction = 1
            elif event.key == pygame.K_a:
                pacman.direction = 2
            elif event.key == pygame.K_d:
                pacman.direction = 3

    pacman.move(GRID)
    died = False
    for ghost in ghosts:
        if pacman.x == ghost.x and pacman.y == ghost.y:
            died = True
            break
    if died:
        break

    if pacman.aggressive():
        pacman.aggressiveness -= 1

    if GRID[pacman.y][pacman.x] == 2:
        GRID[pacman.y][pacman.x] = 0
        score += 1
    elif GRID[pacman.y][pacman.x] == 3:
        GRID[pacman.y][pacman.x] = 0
        pacman.aggressiveness = AGGRESSIVENESS_COOL_DOWN

    draw_grid(screen, pacman, ghosts)
    draw_score(screen, font, score)

    # Flip the display
    pygame.display.flip()
    clock.tick(FPS)

# Done! Time to quit.
pygame.quit()
