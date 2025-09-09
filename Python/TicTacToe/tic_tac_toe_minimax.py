# ============================================================
#   Tic-Tac-Toe Simulator
#   Author : James Stine
#   Institution : Oklahoma State University
#   Date   : September 8, 2025
#   Notes  : AI simulation of Tic-Tac-Toe strategies
# ============================================================

import random
import math

X, O, E = 'X', 'O', ' '

def _percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p  # linear interpolation
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[lo])
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

def _median(xs):
    return _percentile(xs, 0.5)

def new_board():
    return [[E, E, E],
            [E, E, E],
            [E, E, E]]

def empty_cells(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == E]

def get_winner(board):
    # rows
    for r in range(3):
        if board[r][0] == board[r][1] == board[r][2] != E:
            return board[r][0]
    # cols
    for c in range(3):
        if board[0][c] == board[1][c] == board[2][c] != E:
            return board[0][c]
    # diagonals
    if board[0][0] == board[1][1] == board[2][2] != E:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != E:
        return board[0][2]
    return None

def is_full(board):
    return all(E not in row for row in board)

def result(board):
    w = get_winner(board)
    if w:
        return w
    if is_full(board):
        return 'draw'
    return None

def board_key(board):
    return ''.join(''.join(row) for row in board)

def ai_random(board, player, memo=None):
    return random.choice(empty_cells(board)) if empty_cells(board) else None

def minimax(board, player, memo):
    """
    Perfect play. Returns (score, best_move) with score from X's perspective:
      +1 = X wins, 0 = draw, -1 = O wins
    """
    term = result(board)
    if term == X:
        return (1, None)
    if term == O:
        return (-1, None)
    if term == 'draw':
        return (0, None)

    # memoize by (position, side-to-move)
    key = (board_key(board), player)
    if key in memo:
        return memo[key]

    moves = empty_cells(board)
    if player == X:  # maximize
        best = (-2, None)
        for r, c in moves:
            board[r][c] = X
            sc, _ = minimax(board, O, memo)
            board[r][c] = E
            if sc > best[0]:
                best = (sc, (r, c))
                if sc == 1:
                    break
    else:            # minimize
        best = (2, None)
        for r, c in moves:
            board[r][c] = O
            sc, _ = minimax(board, X, memo)
            board[r][c] = E
            if sc < best[0]:
                best = (sc, (r, c))
                if sc == -1:
                    break

    memo[key] = best
    return best

def ai_perfect(board, player, memo=None):
    if memo is None:
        memo = {}
    _, move = minimax(board, player, memo)
    return move

def ai_blundering(board, player, memo=None, blunder_rate=0.10):
    if random.random() < blunder_rate:
        return ai_random(board, player, memo)
    return ai_perfect(board, player, memo)

def play_game(ai_X, ai_O, seed=None, return_log=True):
    """
    ai_X/ai_O: callables(board, player, memo=None) -> (row, col) or None
    Returns: (outcome, moves_log) where outcome in {'X','O','draw'}
    """
    if seed is not None:
        random.seed(seed)

    b = new_board()
    moves_log = []
    to_move = X
    memo = {}  # shared between both players for speed

    for move_idx in range(1, 10):  # max 9 moves
        rc = ai_X(b, X, memo) if to_move == X else ai_O(b, O, memo)
        if rc is None:  # terminal per AI (should coincide with result())
            break

        r, c = rc
        b[r][c] = to_move

        if return_log:
            moves_log.append(
                {'move': move_idx, 'player': to_move, 'row': r, 'col': c})

        out = result(b)
        if out is not None:
            return out, moves_log

        to_move = O if to_move == X else X

    return 'draw', moves_log

def simulate(n_games, ai_X, ai_O, log_first_game=True, seed=None):
    if seed is not None:
        random.seed(seed)

    stats_counts = {'X': 0, 'O': 0, 'draw': 0}
    lengths = []          # store each game length
    opening_counts = {}   # (row, col) -> count
    sample_log = None

    for i in range(n_games):
        out, log = play_game(ai_X, ai_O, seed=None,
                             return_log=log_first_game and (i == 0))
        stats_counts[out] += 1

        if log:
            L = log[-1]['move']
            lengths.append(L)
            first = (log[0]['row'], log[0]['col'])
            opening_counts[first] = opening_counts.get(first, 0) + 1
            if log_first_game and i == 0:
                sample_log = log
        else:
            lengths.append(0)

    # Basic aggregates
    wins_X = stats_counts['X']
    wins_O = stats_counts['O']
    draws = stats_counts['draw']
    avg_len = (sum(lengths) / n_games) if n_games else 0.0

    # Median and quartiles
    median_len = _median(lengths) if lengths else 0.0
    if lengths:
        q1 = _percentile(lengths, 0.25)
        q3 = _percentile(lengths, 0.75)
    else:
        q1 = q3 = 0.0

    # Average score from X’s perspective: +1 win, 0 draw, −1 loss
    avg_score_x = (wins_X - wins_O) / n_games if n_games else 0.0

    # 95% Confidence Interval (CI) for X win rate (normal approximation)
    p = wins_X / n_games if n_games else 0.0
    se = math.sqrt(p * (1 - p) / n_games) if n_games else 0.0
    ci_low = max(0.0, p - 1.96 * se)
    ci_high = min(1.0, p + 1.96 * se)

    return {
        'games': n_games,
        'wins_X': wins_X,
        'wins_O': wins_O,
        'draws': draws,
        'avg_moves': avg_len,
        'median_moves': median_len,
        'q1_moves': q1,
        'q3_moves': q3,
        'avg_score_X': avg_score_x,
        'x_win_rate': p,
        'x_win_ci95': (ci_low, ci_high),
        'openings': opening_counts,
        'sample_log': sample_log
    }

def format_log(log):
    return '\n'.join(f"#{e['move']}: {e['player']} -> (row={e['row']}, col={e['col']})" for e in log)

def print_summary(summary, top_k_openings=5):
    g = summary['games']
    print(f"Games: {g}")
    print(
        f"X wins: {summary['wins_X']}, O wins: {summary['wins_O']}, Draws: {summary['draws']}")

    # Length stats
    print(f"Avg moves: {summary['avg_moves']:.2f} | "
          f"Median: {summary['median_moves']:.1f} | "
          f"IQR: [{summary['q1_moves']:.1f}, {summary['q3_moves']:.1f}]")

    # Score & win rate
    p = summary['x_win_rate']
    lo, hi = summary['x_win_ci95']
    print(f"Avg score (X): {summary['avg_score_X']:.3f} | "
          f"X win rate: {100*p:.1f}% (95% CI: {100*lo:.1f}–{100*hi:.1f}%)")

if __name__ == "__main__":
    # Ask the infamous movie question
    ans = input('Do you want to play a game: CPE-1704-TKS? (y/n) ')
    print("" + "Start".center(72, "-"))
    if ans.strip().lower() not in ('y', 'yes'):
        print("Goodbye.")
        exit()
    print()        

    # 1) Perfect vs Perfect (should draw every time) -- seeds are random
    s1 = simulate(1000, ai_perfect, ai_perfect, log_first_game=True, seed=123)
    print("Perfect vs Perfect:")
    print_summary(s1)
    print()

    # 2) Perfect (X) vs Random (O)
    s2 = simulate(1000, ai_perfect, ai_random,
                  log_first_game=True, seed=456)
    print("Perfect (X) vs Random (O):")
    print_summary(s2)
    print()

    # 3) Random (X) vs Random (O)
    s4 = simulate(1000, ai_random, ai_random,
                  log_first_game=True, seed=456)
    print("Random (X) vs Random (O):")
    print_summary(s4)
    print()    

    # 4) Blundering Perfect (20%) vs Perfect
    s3 = simulate(1000, lambda b, p, memo: ai_blundering(
        b, p, blunder_rate=0.20, memo=memo), ai_perfect, seed=789)
    print("Blundering (20%) Perfect (X) vs Perfect (O):")
    print_summary(s3)
    print()
    
    print("" + "End".center(72, "-"))
    print("A strange game. The only winning move is not to play.")
