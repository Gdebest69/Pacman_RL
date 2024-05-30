import pygame
import random
import heapq
import numpy as np


class Pacman:
    def __init__(self, sprite: str, spawn: tuple[int, int], direction: int = 0) -> None:
        self.sprite = sprite
        self.x = spawn[0]
        self.y = spawn[1]
        self.direction = direction
        self.aggressiveness: int = 0
        self.loaded = False

    def load_sprites(self, size: tuple[int, int]):
        sprite = pygame.transform.scale(pygame.image.load(self.sprite).convert(), size)
        self.sprites = [
            pygame.transform.rotate(sprite, 90),
            pygame.transform.rotate(sprite, -90),
            pygame.transform.rotate(sprite, 180),
            pygame.transform.rotate(sprite, 0),
        ]
        self.loaded = True

    def location(self):
        return self.x, self.y

    def go_up(self, grid):
        self.direction = 0
        if grid[self.y - 1][self.x] != 1:
            self.y -= 1
            return True
        return False

    def go_down(self, grid):
        self.direction = 1
        if grid[self.y + 1][self.x] != 1:
            self.y += 1
            return True
        return False

    def go_left(self, grid):
        self.direction = 2
        if self.x == 0:
            self.x = len(grid[0]) - 1
            return True
        elif grid[self.y][self.x - 1] != 1:
            self.x -= 1
            return True
        return False

    def go_right(self, grid):
        self.direction = 3
        if self.x == len(grid[0]) - 1:
            self.x = 0
            return True
        elif grid[self.y][self.x + 1] != 1:
            self.x += 1
            return True
        return False

    def move(self, grid):
        # up: 0, down: 1, left: 2, right: 3
        if self.direction == 0:
            return self.go_up(grid)
        elif self.direction == 1:
            return self.go_down(grid)
        elif self.direction == 2:
            return self.go_left(grid)
        elif self.direction == 3:
            return self.go_right(grid)

    def aggressive(self):
        return self.aggressiveness > 0


class Ghost:
    def __init__(
        self,
        normal_sprite: str,
        scared_sprite: str,
        spawn: tuple[int, int],
        id: int = 100,
    ):
        self.normal_sprite = normal_sprite
        self.scared_sprite = scared_sprite
        self.spawn = spawn
        self.id = id
        self.x = spawn[0]
        self.y = spawn[1]
        self.loaded = False
        self.death_state = 0

    def load_sprites(self, size):
        self.normal_sprite = pygame.transform.scale(
            pygame.image.load(self.normal_sprite).convert(), size
        )
        self.scared_sprite = pygame.transform.scale(
            pygame.image.load(self.scared_sprite).convert(), size
        )
        self.loaded = True

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
            self.normal_sprite,
            self.scared_sprite,
            (self.spawn[0], self.spawn[1]),
            self.id,
        )

    def go_to_spawn(self):
        self.x, self.y = self.spawn
        self.death_state = 0

    def set_next_location(self, grid, target_location):
        # Define movement directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Helper function to check if a position is valid
        def is_valid_move(x, y):
            if 0 <= x < len(grid[0]) and 0 <= y < len(grid):
                return grid[y][x] != 1
            return False

        # A* algorithm
        def astar(start, target):
            heap = [(0, start)]
            came_from = {}
            cost_so_far = {start: 0}

            while heap:
                current_cost, current_node = heapq.heappop(heap)

                if current_node == target:
                    break

                for dx, dy in directions:
                    next_x, next_y = current_node[0] + dx, current_node[1] + dy

                    if is_valid_move(next_x, next_y):
                        new_cost = current_cost + 1
                        if (
                            next_x,
                            next_y,
                        ) not in cost_so_far or new_cost < cost_so_far[
                            (next_x, next_y)
                        ]:
                            cost_so_far[(next_x, next_y)] = new_cost
                            priority = (
                                new_cost
                                + abs(next_x - target[0])
                                + abs(next_y - target[1])
                            )
                            heapq.heappush(heap, (priority, (next_x, next_y)))
                            came_from[(next_x, next_y)] = current_node

            path = []
            current = target
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        # Call A* algorithm to find the path
        start = (self.x, self.y)
        path = astar(start, target_location)

        # Update ghost's next location
        if len(path) > 1:
            self.x, self.y = path[1]

    def move_to_run_away(self, grid, target_location):
        tx, ty = target_location
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left

        best_distance = 0
        best_move = None

        for dx, dy in directions:
            new_x = self.x + dx
            new_y = self.y + dy

            if not (0 <= new_x < len(grid[0]) and 0 <= new_y < len(grid)):
                continue  # Skip moves outside the grid

            if grid[new_y][new_x] == 1:
                continue  # Skip moves into walls

            distance = abs(new_x - tx) + abs(new_y - ty)
            if distance > best_distance:
                best_distance = distance
                best_move = (new_x, new_y)

        if best_move:
            self.x, self.y = best_move


class PacmanEnv:
    def __init__(
        self,
        grid: list[list[int]],
        *,
        spawn: tuple[int, int] = None,
        pacman_sprite: str = None,
        ghosts: list[Ghost] = None,
        render_mode=None,
        tile_size=25,
        small_dot_radius=2,
        big_dot_radius=10,
        aggressiveness_cooldown=40,
        pacman_id=4,
        fps=7.5,
    ):
        if ghosts is None:
            ghosts = []

        self.start_grid = grid
        self.spawn = spawn
        self.pacman_sprite = pacman_sprite
        self.ghosts = ghosts
        self.width = len(grid[0])
        self.height = len(grid)
        self.tile_size = tile_size
        self.small_dot_radius = small_dot_radius
        self.big_dot_radius = big_dot_radius
        self.aggressiveness_cooldown = aggressiveness_cooldown
        self.pacman_id = pacman_id
        self.fps = fps

        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.screen = None
        self.clock = None

    def _get_obs(self):
        # generate an observation from the current state
        grid = [[tile for tile in row] for row in self.grid]
        if self.pacman.aggressive():
            for ghost in self.current_ghosts:
                grid[ghost.y][ghost.x] = 5
        else:
            for ghost in self.current_ghosts:
                grid[ghost.y][ghost.x] = ghost.id
        grid[self.pacman.y][self.pacman.x] = self.pacman_id
        return grid

    def _pick_empty_tile(self):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        touching_ghost = False
        for ghost in self.ghosts:
            if ghost.x == x and ghost.y == y:
                touching_ghost = True
                break
        while touching_ghost or self.grid[y][x] == 1:  # is in a wall
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            touching_ghost = False
            for ghost in self.ghosts:
                if ghost.x == x and ghost.y == y:
                    touching_ghost = True
                    break
        return x, y

    def reset(
        self, grid: list[list[int]] = None, ghosts: list[Ghost] = None, ignore_win=False
    ):
        if grid is not None:
            self.start_grid = grid
        if ghosts is not None:
            self.ghosts = ghosts
        self.ignore_win = ignore_win
        self.grid = [[tile for tile in row] for row in self.start_grid]
        self.current_ghosts = [ghost.copy() for ghost in self.ghosts]
        """for ghost in self.current_ghosts:
            ghost.x, ghost.y = self._pick_empty_tile()"""
        spawn = self._pick_empty_tile() if self.spawn is None else self.spawn
        self.pacman = Pacman(self.pacman_sprite, spawn, np.random.randint(0, 4))
        self.score = 0
        self.combo = 0

        """for ghost in self.current_ghosts:
            cords = (
                random.randint(1, len(self.grid[0]) - 2),
                random.randint(1, len(self.grid) - 2),
            )
            ghost.x = cords[0]
            ghost.y = cords[1]
            num = random.randint(0, 3)
            if num == 0:
                ghost.y = 1
            elif num == 1:
                ghost.x = len(self.grid[0]) - 2
            elif num == 2:
                ghost.y = len(self.grid) - 2
            elif num == 3:
                ghost.x = 1"""
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame(debug=False)

        return observation

    def step(self, action, debug=False):
        # self.grid[0][0] = random.randint(0, 3)
        pacman_location = self.pacman.location()
        self.pacman.direction = action

        reward = 0 if self.pacman.move(self.grid) else -0.5

        if self.pacman.aggressive():
            self.pacman.aggressiveness -= 1

        if self.grid[self.pacman.y][self.pacman.x] == 2:
            self.grid[self.pacman.y][self.pacman.x] = 0
            self.score += 1
            reward = 1
            # reward = 1 + self.combo * 0.1
            self.combo += 1
        elif self.grid[self.pacman.y][self.pacman.x] == 3:
            self.grid[self.pacman.y][self.pacman.x] = 0
            self.pacman.aggressiveness = self.aggressiveness_cooldown
            reward = 2
        else:
            reward = 0
            self.combo = 0
        observation = self._get_obs()

        died = False
        for ghost in self.current_ghosts:
            if self.pacman.x == ghost.x and self.pacman.y == ghost.y:
                if self.pacman.aggressive():
                    reward = 5
                    ghost.x = -1
                    ghost.y = -1
                    ghost.death_state = 1
                else:
                    died = True
                    reward = -5
                    break
            if ghost.death_state > 0:
                ghost.death_state += 1
            if ghost.death_state > self.aggressiveness_cooldown:
                ghost.go_to_spawn()

        if not died:
            for ghost in self.current_ghosts:
                if ghost.death_state == 0:
                    if self.pacman.aggressive():
                        if random.randint(1, 2) == 1:
                            ghost.move_to_run_away(
                                self._get_obs(), self.pacman.location()
                            )
                    else:
                        if random.randint(1, 3) > 1:
                            ghost.set_next_location(
                                self._get_obs(), self.pacman.location()
                            )
                """if ghost.y == 1 and ghost.x < len(self.grid[0]) - 2:
                    ghost.x += 1
                elif ghost.x == len(self.grid[0]) - 2 and ghost.y < len(self.grid) - 2:
                    ghost.y += 1
                elif ghost.y == len(self.grid) - 1 - 2 and ghost.x > 1:
                    ghost.x -= 1
                elif ghost.x == 1:
                    ghost.y -= 1"""

            died = False
            for ghost in self.current_ghosts:
                if self.pacman.x == ghost.x and self.pacman.y == ghost.y:
                    if self.pacman.aggressive():
                        reward = 5
                        ghost.x = -1
                        ghost.y = -1
                        ghost.death_state = 1
                    else:
                        died = True
                        reward = -5
                        break

        if not self.ignore_win:
            small_dots = 0
            for row in self.grid:
                for tile in row:
                    if tile == 2:
                        small_dots += 1
            if small_dots == 0:
                died = True
                reward = 10

        """
        # this code is to encourage the agent to avoid the ghosts and chanse after them when it can eat them
        if self.current_ghosts:
            weight = self._normalize_distance(
                self._calculate_distance(
                    self.pacman.location(),
                    self._average_point(
                        [ghost.location() for ghost in self.current_ghosts]
                    ),
                ),
                min(len(self.grid), len(self.grid[0])),
            )
            if self.pacman.aggressive():
                reward += weight
            else:
                reward -= weight
        """

        if self.render_mode == "human":
            closed = self._render_frame(debug)
        else:
            closed = False

        return (
            observation,
            reward,
            died,
            pacman_location,
            self.pacman.location(),
            closed,
        )

    def _calculate_distance(self, point1, point2):
        # Calculate Euclidean distance between two points
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _normalize_distance(self, distance, max_distance):
        # Normalize the distance to a range between 0 and 1
        normalized_distance = 1 - (distance / max_distance)
        if normalized_distance < 0:
            return 0
        elif normalized_distance > 1:
            return 1
        else:
            return normalized_distance

    def _average_point(self, points):
        # Initialize sums for x and y coordinates
        sum_x = 0
        sum_y = 0

        # Iterate through the points and accumulate sums
        for point in points:
            sum_x += point[0]
            sum_y += point[1]

        # Calculate the average x and y coordinates
        avg_x = sum_x / len(points)
        avg_y = sum_y / len(points)

        # Return the average point
        return (avg_x, avg_y)

    def _draw_grid(self, grid=None):
        if grid is None:
            grid = self.grid
        COLOR_MAP = {
            0: (0, 0, 0),
            1: (0, 0, 255),
            2: (255, 255, 255),
            3: (255, 200, 200),
        }
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                color = COLOR_MAP[tile]
                if tile == 0 or tile == 1:
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(
                            x * self.tile_size,
                            y * self.tile_size,
                            self.tile_size,
                            self.tile_size,
                        ),
                    )
                elif tile == 2:
                    pygame.draw.rect(
                        self.screen,
                        COLOR_MAP[0],
                        pygame.Rect(
                            x * self.tile_size,
                            y * self.tile_size,
                            self.tile_size,
                            self.tile_size,
                        ),
                    )
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (
                            x * self.tile_size + self.tile_size / 2,
                            y * self.tile_size + self.tile_size / 2,
                        ),
                        self.small_dot_radius,
                    )
                elif tile == 3:
                    pygame.draw.rect(
                        self.screen,
                        COLOR_MAP[0],
                        pygame.Rect(
                            x * self.tile_size,
                            y * self.tile_size,
                            self.tile_size,
                            self.tile_size,
                        ),
                    )
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (
                            x * self.tile_size + self.tile_size / 2,
                            y * self.tile_size + self.tile_size / 2,
                        ),
                        self.big_dot_radius,
                    )

                for ghost in self.current_ghosts:
                    if ghost.death_state == 0:
                        if self.pacman.aggressive():
                            self.screen.blit(
                                ghost.scared_sprite,
                                (ghost.x * self.tile_size, ghost.y * self.tile_size),
                            )
                        else:
                            self.screen.blit(
                                ghost.normal_sprite,
                                (ghost.x * self.tile_size, ghost.y * self.tile_size),
                            )
                if not isinstance(self.pacman.direction, int):
                    print(self.pacman.direction)
                else:
                    self.screen.blit(
                        self.pacman.sprites[self.pacman.direction],
                        (self.pacman.x * self.tile_size, self.pacman.y * self.tile_size),
                    )

    def _draw_score(self, font: pygame.font.Font, score: int):
        text_surface = font.render(f"Score: {score}", False, (255, 255, 255))
        self.screen.blit(text_surface, (0, 0))

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self, debug: bool):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(
                [self.width * self.tile_size, self.height * self.tile_size]
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if not self.pacman.loaded:
            self.pacman.load_sprites((self.tile_size, self.tile_size))
        for ghost in self.current_ghosts:
            if not ghost.loaded:
                ghost.load_sprites((self.tile_size, self.tile_size))

        if self.render_mode == "human":
            self._draw_grid()
            if debug:
                self._draw_debug_grid(
                    pygame.font.SysFont("Comic Sans MS", self.tile_size // 3)
                )
            self._draw_score(pygame.font.SysFont("Comic Sans MS", 30), self.score)

            closed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    closed = True
                    break

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            if self.fps > 0:
                self.clock.tick(self.fps)

            return closed

    def _draw_debug_grid(self, font: pygame.font.Font):
        obs = self._get_obs()
        for y in range(len(obs)):
            for x in range(len(obs[y])):
                text_surface = font.render(f"{x}  {y}", False, (0, 255, 0))
                self.screen.blit(text_surface, (x * self.tile_size, y * self.tile_size))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def best_action(self, q_values):
        valid_actions = []

        if self.pacman.y == 0 or self.grid[self.pacman.y - 1][self.pacman.x] != 1:
            valid_actions.append(0)  # Up
        if (
            self.pacman.y == len(self.grid) - 1
            or self.grid[self.pacman.y + 1][self.pacman.x] != 1
        ):
            valid_actions.append(1)  # Down
        if self.pacman.x == 0 or self.grid[self.pacman.y][self.pacman.x - 1] != 1:
            valid_actions.append(2)  # Left
        if (
            self.pacman.x == len(self.grid[0]) - 1
            or self.grid[self.pacman.y][self.pacman.x + 1] != 1
        ):
            valid_actions.append(3)  # Right

        if not valid_actions:
            return None

        best_action_index = valid_actions[0]
        for i in valid_actions:
            if q_values[i] > q_values[best_action_index]:
                best_action_index = i
        return best_action_index

    def random_action(self):
        valid_actions = []

        if self.pacman.y == 0 or self.grid[self.pacman.y - 1][self.pacman.x] != 1:
            valid_actions.append(0)  # Up
        if (
            self.pacman.y == len(self.grid) - 1
            or self.grid[self.pacman.y + 1][self.pacman.x] != 1
        ):
            valid_actions.append(1)  # Down
        if self.pacman.x == 0 or self.grid[self.pacman.y][self.pacman.x - 1] != 1:
            valid_actions.append(2)  # Left
        if (
            self.pacman.x == len(self.grid[0]) - 1
            or self.grid[self.pacman.y][self.pacman.x + 1] != 1
        ):
            valid_actions.append(3)  # Right

        if not valid_actions:
            return None

        return random.choice(valid_actions)


def generate_grid(grid_size: tuple[int, int]):
    def check_holes(grid):
        def count_space(grid, x, y, places_checked: list[tuple[int, int]]):
            len_x = len(grid[0])
            len_y = len(grid)
            if (
                0 <= x < len_x
                and 0 <= y < len_y
                and grid[y][x] != 1
                and (x, y) not in places_checked
            ):
                places_checked.append((x, y))
                return (
                    1
                    + count_space(grid, x + 1, y, places_checked)
                    + count_space(grid, x - 1, y, places_checked)
                    + count_space(grid, x, y + 1, places_checked)
                    + count_space(grid, x, y - 1, places_checked)
                )
            else:
                return 0

        empty_spaces = 0
        for row in grid:
            for tile in row:
                if tile != 1:
                    empty_spaces += 1
        checked_x = random.randint(0, x - 1)
        checked_y = random.randint(0, y - 1)
        while grid[checked_y][checked_x] == 1:
            checked_x = random.randint(0, x - 1)
            checked_y = random.randint(0, y - 1)
        return count_space(grid, checked_x, checked_y, []) == empty_spaces

    x = grid_size[1]
    y = grid_size[0]
    length = x * y
    grid = [[2 for _ in range(x)] for _ in range(y)]
    walls = random.randint(length // 4, length // 4 * 3)
    for _ in range(walls):
        wall_x = random.randint(0, x - 1)
        wall_y = random.randint(0, y - 1)
        while grid[wall_y][wall_x] == 1:
            wall_x = random.randint(0, x - 1)
            wall_y = random.randint(0, y - 1)
        grid[wall_y][wall_x] = 1
    while not check_holes(grid):
        grid = [[2 for _ in range(x)] for _ in range(y)]
        walls = random.randint(length // 4, length // 4 * 3)
        for _ in range(walls):
            wall_x = random.randint(0, x - 1)
            wall_y = random.randint(0, y - 1)
            while grid[wall_y][wall_x] == 1:
                wall_x = random.randint(0, x - 1)
                wall_y = random.randint(0, y - 1)
            grid[wall_y][wall_x] = 1
    grid = (
        [[1 for _ in range(x + 2)]]
        + [[1] + row + [1] for row in grid]
        + [[1 for _ in range(x + 2)]]
    )
    return grid


from collections import deque


def find_nearest_point(grid, pacman_location):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

    # Initialize queue for BFS
    queue = deque([(pacman_location, 0)])  # (position, distance)
    visited = set()  # Keep track of visited positions

    while queue:
        (x, y), distance = queue.popleft()

        # Check if the current position is a point
        if grid[y][x] == 2:
            return distance

        # Explore adjacent cells
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (
                0 <= new_x < len(grid[0])
                and 0 <= new_y < len(grid)
                and grid[new_y][new_x] != 1
                and (new_x, new_y) not in visited
            ):
                queue.append(((new_x, new_y), distance + 1))
                visited.add((new_x, new_y))

    # If no point is found, return -1
    return -1


def get_next_move(grid, pacman_location):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    min_distance = float("inf")
    next_move = None

    for i, (dx, dy) in enumerate(directions):
        new_x, new_y = pacman_location[0] + dx, pacman_location[1] + dy
        if (
            0 <= new_x < len(grid[0])
            and 0 <= new_y < len(grid)
            and grid[new_y][new_x] != 1
        ):
            distance = find_nearest_point(grid, (new_x, new_y))
            if distance != -1 and distance < min_distance:
                min_distance = distance
                next_move = i

    return next_move


def get_tiny_obs(grid, pacman_location, r: int):
    pacman_x, pacman_y = pacman_location
    tiny_grid = []
    for y in range(pacman_y - r, pacman_y + r + 1):
        if not 0 <= y < len(grid):
            tiny_grid.append([1] * (r * 2 + 1))
        else:
            row = []
            for x in range(pacman_x - r, pacman_x + r + 1):
                if not 0 <= x < len(grid[0]):
                    row.append(1)
                else:
                    row.append(grid[y][x])
            tiny_grid.append(row)
    # artificial points
    has_point = False
    for row in tiny_grid:
        if 2 in row:
            has_point = True
            break
    if not has_point:
        direction = get_next_move(grid, pacman_location)
        if direction == 0:  # up
            tiny_grid[0][r // 2] = 2
        elif direction == 1:  # down
            tiny_grid[-1][r // 2] = 2
        elif direction == 2:  # left
            tiny_grid[r // 2][0] = 2
        elif direction == 3:  # right
            tiny_grid[r // 2][-1] = 2

    return tiny_grid
