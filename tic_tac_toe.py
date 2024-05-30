import pygame
import random
import typing


def hash_obs(obs: list[int]):
    """
    Hashes the observation so it can be used as a key
    """
    return tuple(obs)


def random_action(obs: list[int]):
    """
    Returns a random action in a specific observation
    """
    return random.choice([i for i in range(len(obs)) if obs[i] == 0])


def best_action(q_actions: list[float], obs: list[int]):
    """
    Returns the best action in a specific observation from a Q actions list
    """
    best_action_index = 0
    while obs[best_action_index] != 0:
        best_action_index += 1
    for i in range(best_action_index + 1, len(obs)):
        if obs[i] == 0 and q_actions[i] > q_actions[best_action_index]:
            best_action_index = i
    return best_action_index


def epsilon_greedy_action(epsilon: float, q_actions: list[float], obs: list[int]):
    """
    Returns an epsilon greedy action based on a specific observation and a Q actions list
    """
    if random.random() < epsilon:
        # explore
        return random_action(obs)
    # exploit
    return best_action(q_actions, obs)


def game_state(obs):
    """
    Returns the state of the game in a specfic observation\n
    `0`: The game is still going\n
    `1`: The environment won\n
    `2`: the agent won\n
    `3`: Tie
    """
    for i in range(3):
        # Check rows
        if obs[i * 3] == obs[i * 3 + 1] == obs[i * 3 + 2] and obs[i * 3] != 0:
            return obs[i * 3]

        # Check columns
        if obs[i] == obs[i + 3] == obs[i + 6] and obs[i] != 0:
            return obs[i]

    # Check diagonals
    if obs[0] == obs[4] == obs[8] and obs[0] != 0:
        return obs[0]
    if obs[2] == obs[4] == obs[6] and obs[2] != 0:
        return obs[2]

    # Check for tie
    for i in range(len(obs)):
        if obs[i] == 0:
            return 0

    # No one won and there is no empty space, so it must be a tie
    return 3


def win_action(obs: list[int], player: typing.Literal[1, 2]):
    """
    If the `player` can win in 1 move, the winning action will be returned.
    Will return `-1` if there is no winning action.
    """
    temp_obs = obs.copy()
    for i in range(len(temp_obs)):
        if temp_obs[i] == 0:
            temp_obs[i] = player
            if game_state(temp_obs) == player:
                return i
            temp_obs[i] = 0
    return -1


def block_action(obs: list[int], player: typing.Literal[1, 2]):
    """
    If the `player` can block the opponent, preventing him from winning, the blocking action will be returned.
    Will return `-1` if there is no blocking action.
    """
    opponent = 1 if player == 2 else 2
    action = win_action(obs, opponent)
    if action != -1:  # Opponent can win, so we can block him
        return action
    return -1


class TicTacToeEnv:
    """
    0 = empty\n
    1 = x placed by env\n
    2 = o placed be agent
    """

    def __init__(
        self,
        render: bool = False,
        fps: int = -1,
        human_env: bool = False,
        auto_block: bool = False,
        auto_win: bool = False,
        env_policy=random_action,
        win_reward: int = 10,
        lose_reward: int = -10,
        tie_reward: int = 1,
        tile_size: int = 100,
        shape_size: int = 75,
    ) -> None:
        """
        `render`: If set to `True` the game will be rendered visually.\n
        `fps`: The game will run with that amount of FPS, useful for when `render` is set to `True`.
        Passing a negative amount means that there is no delay between frames.\n
        `human_env`: If set to `True` the actions of the environment will be made manually,
        meaning that you have to click at the tile that you want the environment to use
        (useful when you want to test an agent against a human).
        Setting this to `False` means that the environment will make actions based on its policy.\n
        `auto_block`: If set to `True` the environment will automatically block the agent when it can.\n
        `auto_win`: If set to `True` the environment will automatically win when it can.\n
        `env_policy`: The policy that the environment uses.
        It should be a function that gets an observation from the environment and returns a VALID action.\n
        Note that `auto_block` and `auto_win` will be prioritized over the environment's policy.\n
        `win_reward`: The reward that gets returned when the agent wins.\n
        `lose_reward`: The reward that gets returned when the agent loses.\n
        `tie_reward`: The reward that gets returned when the game ends in a tie.\n
        `tile_size`: The width and height in pixels of each of the 9 tiles.\n
        `shape_size`: The width and height in pixels of the X and O symbols.
        """
        self.grid = [[0 for _ in range(3)] for _ in range(3)]
        self.render = render
        self.fps = fps
        if human_env and not render:
            raise ValueError(
                "render needs to be set to True for human_env to be set to True"
            )
        self.human_env = human_env
        self.auto_block = auto_block
        self.auto_win = auto_win
        self.env_policy = env_policy
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.tie_reward = tie_reward
        self.tile_size = tile_size
        self.shape_size = shape_size
        self.font = None
        self.screen = None
        self.clock = None

    def _get_obs(self):
        grid = []
        for row in self.grid:
            grid += row
        return grid

    def _env_step(self):
        if self.human_env:
            action = self.wait_for_human_action()
            y = action // 3
            x = action % 3
            while self.grid[y][x] != 0:
                action = self.wait_for_human_action()
                y = action // 3
                x = action % 3
        else:
            obs = self._get_obs()
            action = -1
            if self.auto_win:
                action = win_action(obs, 1)
            if self.auto_block and action == -1:
                action = block_action(obs, 1)
            if action == -1:
                action = self.env_policy(obs)
        y = action // 3
        x = action % 3
        if self.grid[y][x] != 0:
            raise ValueError(
                "The environment didn't choose an empty tile. Make sure that env_policy returns a valid action."
            )
        self.grid[y][x] = 1
        gamestate = game_state(self._get_obs())
        obs = self._get_obs()

        if self.render:
            self._render()

        if gamestate == 0:
            return obs, 0, False
        elif gamestate == 1:
            return obs, self.lose_reward, True
        elif gamestate == 2:
            return obs, self.win_reward, True
        elif gamestate == 3:
            return obs, self.tie_reward, True

    def reset(
        self,
        env_starts=True,
        auto_block: bool = None,
        auto_win: bool = None,
        env_policy=None,
    ):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        if auto_block is not None:
            self.auto_block = auto_block
        if auto_win is not None:
            self.auto_win = auto_win
        if env_policy is not None:
            self.env_policy = env_policy
        self.grid = [[0 for _ in range(3)] for _ in range(3)]
        if self.human_env:
            self._render()
        if env_starts:
            self._env_step()
        return self._get_obs()

    def step(self, action: int):
        """
        Run one timestep of the environment's dynamics.\n
        When end of episode is reached, you are responsible for calling reset to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, terminated).\n
        `action`: A number between 0 and 8 that represents the location of an empty tile.
        """
        if not 0 <= action < 9:
            raise ValueError("Action must be between 0 and 8")

        y = action // 3
        x = action % 3
        if self.grid[y][x] != 0:
            raise ValueError("Tile is not empty")

        self.grid[y][x] = 2
        gamestate = game_state(self._get_obs())
        obs = self._get_obs()

        if self.render:
            self._render()

        if gamestate == 0:
            return self._env_step()
        elif gamestate == 1:
            return obs, self.lose_reward, True
        elif gamestate == 2:
            return obs, self.win_reward, True
        elif gamestate == 3:
            return obs, self.tie_reward, True

    def _render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.tile_size * 3, self.tile_size * 3)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("Comic Sans MS", self.shape_size)

        self.screen.fill((0, 0, 0))
        for y in range(1, 3):
            pygame.draw.line(
                self.screen,
                (0, 0, 255),
                (0, y * self.tile_size),
                (self.tile_size * 3, y * self.tile_size),
                self.tile_size // 10,
            )
        for x in range(1, 3):
            pygame.draw.line(
                self.screen,
                (0, 0, 255),
                (x * self.tile_size, 0),
                (x * self.tile_size, self.tile_size * 3),
                self.tile_size // 10,
            )
        for y in range(3):
            for x in range(3):
                grid_value = self.grid[y][x]
                text = ""
                if grid_value == 1:
                    text = "x"
                elif grid_value == 2:
                    text = "o"
                text_surface = self.font.render(text, False, (255, 255, 255))
                self.screen.blit(
                    text_surface,
                    (
                        x * self.tile_size + (self.tile_size - self.shape_size) // 2,
                        y * self.tile_size,
                    ),
                )

        pygame.display.update()
        if self.fps > 0:
            self.clock.tick(self.fps)

    def close(self):
        """
        Closes the display if it was used
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def wait_for_human_action(self) -> int:
        """
        Return an action which is made by clicking a tile.
        Can only be used if `render` is set to `True`.
        """
        if not self.render:
            raise ValueError(
                "render must be set to True in order to call this function"
            )
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise KeyboardInterrupt("The game window was closed")
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    grid_x = x // self.tile_size
                    grid_y = y // self.tile_size
                    return grid_y * 3 + grid_x

            if self.fps > 0:
                self.clock.tick(self.fps)
