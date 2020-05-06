"""Classic minesweeper game made fully in Python language."""
import typing as t
import enum
import threading
import importlib

import numpy as np
import matplotlib.pyplot as plt

SYMB_COLORS = {
    "F": (1.00, 1.00, 1.00),
    "M": (0.87, 0.08, 0.40),
    "1": (0.00, 0.00, 1.00),
    "2": (0.00, 1.00, 0.00),
    "3": (1.00, 0.00, 0.00),
    "4": (0.09, 0.01, 0.35),
    "5": (0.43, 0.01, 0.15),
    "6": (0.92, 0.31, 0.02),
    "7": (0.01, 0.30, 0.04),
    "8": (0.18, 0.02, 0.50),
}
"""Colors of each number, starting on 1."""

GAME_OVER_COLOR = (0.720, 0.630, 0.760)

GRID_COLOR = (0.96, 0.96, 0.96)

CONFIG = {
    "easy": (9, 9, 10),
    "medium": (16, 16, 40),
    "expert": (30, 16, 99),
}
"""Default game configurations."""

CMAP_DEFAULT = "gist_stern"
CMAP_GAME_OVER = "RdGy"
CMAP_WIN = "spring"


class states_id(enum.IntEnum):
    """Apply semantic to states ID of each cell."""
    CLOSED = 0
    MARKED = 1
    OPEN = 2
    EXPLODED = 2


class _Timer:
    """Count game seconds."""
    def __init__(self, time: int, ax: t.Any):
        self.time = time
        self.ax = ax
        self.time_counter = 0
        self._timer = None
        self.started = False

    def inc_time(self):
        self.time_counter += 1
        self.start()

    def start(self):
        self.started = True
        self._timer = threading.Timer(self.time, self.inc_time)
        self._timer.start()

    def stop(self):
        if self._timer is not None:
            self._timer.cancel()


_mine_threshold = 9
_adj_inds_x, _adj_inds_y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
_adj_inds_x = _adj_inds_x.astype(np.int8).ravel()
_adj_inds_y = _adj_inds_y.astype(np.int8).ravel()
_marked_mines_count = 0
_openned_tiles = 0
_game_is_over = False
_restart_game = True
_player_wins = False


def is_mine(val: int) -> bool:
    """Return whether a given position (x, y) is a mine."""
    return val >= _mine_threshold


def is_empty(val: int) -> bool:
    """Return whether a given position (x, y) is empty."""
    return val == 0


def is_closed(state: int) -> bool:
    """Return whether a given position (x, y) is closed."""
    return state == states_id.CLOSED


def is_open(state: int) -> bool:
    """Return whether a given position (x, y) is open."""
    return state == states_id.OPEN


def is_marked(state: int) -> bool:
    """Return whether a given position (x, y) is marked."""
    return state == states_id.MARKED


def is_valid_coord(board: np.ndarray, x: int, y: int) -> bool:
    """Check if given coordinate is in the game boundaries."""
    return (1 <= x < board.shape[1] - 1) and (1 <= y < board.shape[0] - 1)


def flag_spot(states: np.ndarray, x: int, y: int,
              annotations: np.ndarray) -> None:
    """Put a flag in (x, y) coordinate."""
    global _marked_mines_count
    x = x + 1
    y = y + 1

    if is_marked(states[y, x]):
        states[y, x] = states_id.CLOSED
        annotations[y, x] = ""
        _marked_mines_count -= 1

    elif is_closed(states[y, x]):
        states[y, x] = states_id.MARKED
        annotations[y, x] = "F"
        _marked_mines_count += 1


def check_player_win(tiles_to_open: int) -> bool:
    """Check whether the player won."""
    print(_openned_tiles, tiles_to_open)
    if not _game_is_over and _openned_tiles == tiles_to_open:
        global _player_wins
        _player_wins = True

    return _player_wins


def dfs_open(board: np.ndarray, states: np.ndarray, x: int, y: int,
             annotations: np.ndarray) -> None:
    """Open empty positions iteratively."""
    if not 0 <= x < board.shape[1] - 2:
        return

    if not 0 <= y < board.shape[0] - 2:
        return

    x = x + 1
    y = y + 1

    if not is_closed(states[y, x]):
        return

    pos = [(y, x)]

    global _openned_tiles

    while pos:
        cur_y, cur_x = pos.pop()

        if not is_closed(states[cur_y, cur_x]):
            continue

        states[cur_y, cur_x] = states_id.OPEN
        _openned_tiles += 1

        if not is_empty(board[cur_y, cur_x]):
            annotations[cur_y, cur_x] = str(board[cur_y, cur_x])

        else:
            for neigh_y, neigh_x in zip(cur_y + _adj_inds_y,
                                        cur_x + _adj_inds_x):
                if (is_valid_coord(board, x=neigh_x, y=neigh_y)
                        and is_closed(states[neigh_y, neigh_x])):
                    pos.append((neigh_y, neigh_x))


def _set_pos_val(board: np.ndarray, x: int, y: int, val: int,
                 annotations: np.ndarray) -> None:
    """Set value at given position."""
    x = x + 1
    y = y + 1

    # If value corresponds to a mine, increment neighborhood
    inc = int(is_mine(val))

    # If current position is a mine, decrement neighborhood
    if is_mine(board[y, x]):
        inc -= 1

    _inds_y = y + _adj_inds_y
    _inds_x = x + _adj_inds_x

    board[_inds_y, _inds_x] = np.int8(board[_inds_y, _inds_x] + inc)
    board[y, x] = val

    # Note: solve annotations
    _inds_mines = is_mine(board[_inds_y, _inds_x])
    _inds_nzero = np.logical_and(~_inds_mines, board[_inds_y, _inds_x] > 0)
    _inds_zeros = board[_inds_y, _inds_x] == 0

    _inds_num_y = y + _adj_inds_y[_inds_nzero]
    _inds_num_x = x + _adj_inds_x[_inds_nzero]
    annotations[_inds_num_y, _inds_num_x] = board[_inds_num_y,
                                                  _inds_num_x].astype(str)

    annotations[y + _adj_inds_y[_inds_mines],
                x + _adj_inds_x[_inds_mines]] = "M"
    annotations[y + _adj_inds_y[_inds_zeros],
                x + _adj_inds_x[_inds_zeros]] = ""


def place_mines(board: np.ndarray, x: t.Union[np.ndarray, int],
                y: t.Union[np.ndarray, int], annotations: np.ndarray) -> None:
    """Place mines in the given positions on the board."""
    if np.isscalar(x):
        x = np.asarray(x, dtype=int)

    if np.isscalar(y):
        y = np.asarray(y, dtype=int)

    if not np.all(np.logical_and(0 <= x, x < board.shape[1] - 2)):
        raise ValueError(f"'x' must be 0 <= {x} < {board.shape[1] - 2}")

    if not np.all(np.logical_and(0 <= y, y < board.shape[0] - 2)):
        raise ValueError(f"'y' must be 0 <= {y} < {board.shape[0] - 2}")

    for cur_x, cur_y in zip(x, y):
        _set_pos_val(board,
                     x=cur_x,
                     y=cur_y,
                     val=2 * _mine_threshold,
                     annotations=annotations)


def handle_first_click(board: np.ndarray,
                       x: int,
                       y: int,
                       annotations: np.ndarray,
                       random_state: t.Optional[int] = None) -> None:
    """Handle first click, avoiding hitting a mine."""
    if random_state is not None:
        np.random.seed(random_state)

    new_y, new_x = y, x

    while is_mine(board[new_y + 1, new_x + 1]):
        # Note: if it is a mine, then moves mine to a new random
        # position.
        new_y = np.random.randint(board.shape[0] - 2)
        new_x = np.random.randint(board.shape[1] - 2)

    if new_y != y or new_x != x:
        _set_pos_val(board, x=x, y=y, val=0, annotations=annotations)
        _set_pos_val(board,
                     x=new_x,
                     y=new_y,
                     val=2 * _mine_threshold,
                     annotations=annotations)


def init_board(
    width: int,
    height: int,
    num_mines: int,
    annotations: np.ndarray,
    random_state: t.Optional[int] = None,
) -> t.Union[np.ndarray, np.ndarray]:
    """Init a (height, width) board with given number of mines."""
    height = int(height)
    width = int(width)
    num_mines = int(num_mines)

    if height <= 0:
        raise ValueError(f"'height' must be positive (got {height}).")

    if width <= 0:
        raise ValueError(f"'width' must be positive (got {width}).")

    if num_mines <= 0:
        raise ValueError(f"'num_mines' must be positive (got {num_mines}).")

    # Note: the first and last columns and rows are 'virtual', i.e.
    # is does not make part of the actual board, and it is used only
    # to simplify the code.
    board, states = np.zeros((2, 2 + height, 2 + width), dtype=np.int8)

    if random_state is not None:
        np.random.seed(random_state)

    mines_inds = np.random.choice(a=height * width,
                                  size=num_mines,
                                  replace=False)

    place_mines(board,
                x=mines_inds % width,
                y=mines_inds // width,
                annotations=annotations)

    return board, states


def _draw_plot(board: np.ndarray,
               states: np.ndarray,
               annotations: np.ndarray,
               num_mines: int,
               random_state: t.Optional[int] = None) -> t.Tuple[t.Any]:
    """Draw game window."""
    def _canonical_axis(ax):
        """Build a standardized axis."""
        cmap = CMAP_DEFAULT

        if _game_is_over: cmap = CMAP_GAME_OVER
        elif _player_wins: cmap = CMAP_WIN

        ax.matshow(states[1:-1, 1:-1], picker=1, cmap=cmap)

        ax.grid(alpha=1, color=GRID_COLOR, linewidth=2, which="both")
        ax.set_xticks(np.arange(-0.5, board.shape[1] - 1.5, 1))
        ax.set_yticks(np.arange(-0.5, board.shape[0] - 1.5, 1))
        ax.tick_params(left=False,
                       top=False,
                       bottom=False,
                       labeltop=False,
                       labelleft=False)

    def _fill_anotations(annotations: np.ndarray):
        """Annotate the game fields."""
        for y, x in zip(*np.nonzero(annotations)):
            if not is_closed(states[y, x]):
                symb = annotations[y, x]

                if _game_is_over and symb != "M":
                    col = GAME_OVER_COLOR

                else:
                    col = SYMB_COLORS[symb]

                ax.annotate(symb, (x - 1, y - 1),
                            color=col,
                            textcoords="offset points",
                            xytext=(0, 0),
                            size=16,
                            ha="center",
                            va="center")

    def _mouse_event_onpick(event):
        """Handle user mouse clicks."""
        if _game_is_over or _player_wins:
            global _restart_game
            _restart_game = True
            plt.close(fig)
            return

        mouse = event.mouseevent

        # Note: 'y' and 'x' are corrected for matplotlib
        # coordinates, but not for the game board 'virtual' offset.
        # Hence, adding +1 to both variables is necessary to
        # correctly index the board coordinate.
        y, x = int(mouse.ydata + 0.5), int(mouse.xdata + 0.5)

        if not timer.started:
            timer.start()

            if mouse.button == 1:
                handle_first_click(board=board,
                                   x=x,
                                   y=y,
                                   annotations=annotations,
                                   random_state=random_state)

        if mouse.button != 2 and is_open(states[y + 1, x + 1]):
            return

        if mouse.button == 1 and is_closed(states[y + 1, x + 1]):
            ax.texts = []
            ax.cla()

            if is_mine(board[y + 1, x + 1]):
                game_over(board=board, states=states, annotations=annotations)
                timer.stop()

            else:
                dfs_open(board=board,
                         states=states,
                         x=x,
                         y=y,
                         annotations=annotations)

            check_player_win(tiles_to_open)
            _canonical_axis(ax)
            _fill_anotations(annotations)

        elif mouse.button == 2:
            ax.texts = []
            ax.cla()

            for cur_y, cur_x in zip(y + _adj_inds_y, x + _adj_inds_x):
                if is_closed(states[cur_y + 1, cur_x + 1]):
                    if is_mine(board[cur_y + 1, cur_x + 1]):
                        game_over(board=board,
                                  states=states,
                                  annotations=annotations)
                        timer.stop()

                    elif not _game_is_over:
                        dfs_open(board=board,
                                 states=states,
                                 x=cur_x,
                                 y=cur_y,
                                 annotations=annotations)

            check_player_win(tiles_to_open)
            _canonical_axis(ax)
            _fill_anotations(annotations)

        elif mouse.button == 3:
            ax.texts = []
            ax.cla()
            flag_spot(states=states, x=x, y=y, annotations=annotations)
            _canonical_axis(ax)
            _fill_anotations(annotations)

        ax.set_title(
            f"Time counter: {timer.time_counter:03d}s "
            f"{'(Game is over! Click anywhere to restart.)' if _game_is_over else ''}"
            f"{'(Good job! Click anywhere to restart.)' if _player_wins else ''}"
        )

        ax.set_xlabel(
            f"Mines remaining: {num_mines - _marked_mines_count:02d}")
        plt.draw()

    tiles_to_open = (board.shape[0] - 2) * (board.shape[1] - 2) - num_mines
    fig = plt.figure()
    fig.suptitle("Minesweepyr")
    ax = fig.add_subplot(111)
    timer = _Timer(1, ax=ax)
    _canonical_axis(ax)
    fig.canvas.mpl_connect("pick_event", _mouse_event_onpick)

    return fig, ax, timer


def start_game(width: int,
               height: int,
               num_mines: int,
               random_state: t.Optional[int] = None) -> None:
    """Start a new game."""
    global _marked_mines_count
    global _game_is_over
    global _restart_game
    global _player_wins
    global _openned_tiles

    _restart_game = False
    _marked_mines_count = 0
    _openned_tiles = 0
    _game_is_over = False
    _player_wins = False

    annotations = np.zeros((2 + height, 2 + width), dtype="U1")
    board, states = init_board(width=width,
                               height=height,
                               num_mines=num_mines,
                               annotations=annotations,
                               random_state=random_state)

    fig, ax, timer = _draw_plot(board=board,
                                states=states,
                                annotations=annotations,
                                num_mines=num_mines,
                                random_state=random_state)

    try:
        plt.show()

    except Exception:
        pass

    finally:
        timer.stop()
        importlib.reload(plt)


def game_over(board: np.ndarray, states: np.ndarray,
              annotations: np.ndarray) -> None:
    global _game_is_over
    _game_is_over = True
    inds_mines = is_mine(board)
    states[inds_mines] = states_id.EXPLODED
    annotations[inds_mines] = "M"


def _test() -> None:
    import sys

    if len(sys.argv) <= 1:
        print("usage: python", sys.argv[0], "(easy|medium|expert)")
        exit(1)

    config = CONFIG.get(sys.argv[1])

    if config is None:
        print(f"Unknown difficult '{sys.argv[1]}'. Choose one "
              "in {easy, medium, expert}.")
        exit(2)

    while _restart_game:
        start_game(*config)


if __name__ == "__main__":
    _test()
