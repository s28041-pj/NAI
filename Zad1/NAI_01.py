#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Isolation

A minimal implementation of the Isolation game in Python.

Rules:
See the README.md file in this repository.
https://github.com/s28041-pj/NAI/blob/main/Zad1/readme.md

Authors:
- Mikołaj Gurgul, Łukasz Aleksandrowicz

## How to run the game:

1. Install Python (>=3.8)
2. Clone this repository
3. Run the game:
    ```bash
    python NAI_01.py
    ```
4. Enter the number of rows and columns (max 6x6)
5. Choose game mode (human-human, human-AI, or AI-AI)
6. Enter AI search depth (max 4)
7. Enjoy the game!

The AI will respond automatically.
"""

from typing import List, Tuple, Optional
import math
import time

# Type aliases
Board = List[List[str]]
Pos = Tuple[int, int]

# Constants
EMPTY = '·'   # Empty cell
BLOCK = '#'   # Blocked cell
P1 = 'A'      # Player 1 symbol (logic only)
P2 = 'B'      # Player 2 symbol (logic only)

# Colored markers (for terminal display)
BLUE = "🔵"
RED = "🔴"

# 8 possible directions for movement (king-like moves)
DIRS = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)]


def make_board(r: int, c: int) -> Board:
    """
    Create a new empty board.

    Args:
        r (int): number of rows
        c (int): number of columns

    Returns:
        Board: a 2D list filled with EMPTY symbols
    """
    return [[EMPTY for _ in range(c)] for _ in range(r)]


def in_bounds(b: Board, p: Pos) -> bool:
    """
    Check if a position is within the board boundaries.
    """
    r, c = p
    return 0 <= r < len(b) and 0 <= c < len(b[0])


def moves(b: Board, p: Pos, opp: Optional[Pos]) -> List[Pos]:
    """
    Compute all possible legal moves for a given position.

    Args:
        b (Board): current board state
        p (Pos): current position of the player
        opp (Pos): position of the opponent

    Returns:
        List[Pos]: all valid moves
    """
    if p is None:
        return []
    out = []
    for dr, dc in DIRS:
        np = (p[0] + dr, p[1] + dc)
        # Move must be inside board, on an empty square, and not occupied by the opponent
        if in_bounds(b, np) and b[np[0]][np[1]] == EMPTY and np != opp:
            out.append(np)
    return out


def apply_move(b: Board, from_p: Pos, to_p: Pos) -> Tuple[Board, Pos]:
    """
    Apply a move on the board: mark the previous cell as BLOCK and return the new board.

    Args:
        b (Board): current board
        from_p (Pos): current player position
        to_p (Pos): new player position

    Returns:
        Tuple[Board, Pos]: updated board and new player position
    """
    nb = [row[:] for row in b]
    if from_p is not None:
        nb[from_p[0]][from_p[1]] = BLOCK
    return nb, to_p


def print_board(b: Board, a: Optional[Pos], bb: Optional[Pos]):
    """
    Print the current board state in a formatted layout with lettered columns and numbered rows.

    Args:
        b (Board): current board
        a (Pos): player A position
        bb (Pos): player B position
    """
    cols = len(b[0])
    print("\n   " + "  ".join(chr(i + 65) for i in range(cols)))  # A, B, C, ...
    for i, row in enumerate(b):
        line = []
        for j, cell in enumerate(row):
            if (i, j) == a:
                line.append(BLUE)
            elif (i, j) == bb:
                line.append(RED)
            else:
                line.append(cell)
        print(f"{i + 1:2} " + " ".join(line))
    print()


def heuristic(b: Board, my: Pos, opp: Pos) -> int:
    """
    Basic heuristic function: difference between available moves of player and opponent.

    Returns:
        int: positive if the current player has an advantage
    """
    return len(moves(b, my, opp)) - len(moves(b, opp, my))


def negamax(b: Board, my: Pos, opp: Pos, depth: int) -> Tuple[int, Optional[Pos]]:
    """
    Negamax recursive search algorithm for AI decision-making.

    Args:
        b (Board): current board
        my (Pos): current player position
        opp (Pos): opponent position
        depth (int): search depth

    Returns:
        Tuple[int, Optional[Pos]]: (best heuristic value, best move)
    """
    legal = moves(b, my, opp)
    if not legal:
        return -9999, None  # No legal moves → loss
    if depth == 0:
        return heuristic(b, my, opp), None

    best_v = -math.inf
    best_m = None

    for m in legal:
        nb, new_my = apply_move(b, my, m)
        val, _ = negamax(nb, opp, new_my, depth - 1)
        val = -val  # Negamax symmetry
        if val > best_v:
            best_v = val
            best_m = m
    return int(best_v), best_m


def ai_move(b: Board, my: Pos, opp: Pos, depth: int = 2) -> Optional[Pos]:
    """
    Wrapper for negamax to get the best AI move.
    """
    _, mv = negamax(b, my, opp, depth)
    return mv


def parse_token_pair(s: str, rows: int, cols: int) -> Optional[Pos]:
    """
    Parse a user input like "A1", "1A", "r c", "r,c", or "r;c" into (row, col).

    Args:
        s (str): user input string
        rows (int): number of rows
        cols (int): number of columns

    Returns:
        Optional[Pos]: (row, col) tuple or None if invalid
    """
    s = s.strip().lower().replace(',', ' ').replace(';', ' ')
    parts = s.split()

    # Case 1: two numeric tokens
    if len(parts) == 2:
        try:
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            if 0 <= r < rows and 0 <= c < cols:
                return (r, c)
        except Exception:
            return None

    # Case 2: alphanumeric (like "a1" or "1a")
    letters = ''.join(ch for ch in s if ch.isalpha())
    digits = ''.join(ch for ch in s if ch.isdigit())

    if digits and letters:
        try:
            row = int(digits) - 1
            col = ord(letters[0]) - ord('a')
            if 0 <= row < rows and 0 <= col < cols:
                return (row, col)
        except Exception:
            return None
    return None


def interactive_move_input(prompt: str, rows: int, cols: int) -> Optional[Pos]:
    """
    Get a player's move from input and parse it into a valid position.

    Returns:
        Optional[Pos]: player move or None if quitting
    """
    s = input(prompt).strip()
    if s.lower() == 'q':
        return None

    pos = parse_token_pair(s, rows, cols)
    if pos is not None:
        return pos
    return None


def play():
    """
    Main game loop: handles setup, turns, AI logic, and win conditions.
    """
    print("*** Isolation ***")

    # Board setup
    rows = input("Rows (max 6, default 4): ").strip() or "4"
    cols = input("Columns (max 6, default 4): ").strip() or "4"
    try:
        R, C = min(6, max(2, int(rows))), min(6, max(2, int(cols)))
    except:
        R, C = 4, 4

    b = make_board(R, C)
    a_pos = (0, 0)
    b_pos = (R - 1, C - 1)

    # Game mode selection
    mode = input("Mode: human-human (h), human-AI (a), AI-AI (aa)? [a]: ").strip().lower() or 'a'

    # AI depth configuration
    depth1 = input("AI depth (max 4, default 2): ").strip() or "2"
    depth2 = input("AI depth for second AI (if AI-AI): ").strip() or depth1
    try:
        depth1 = min(4, max(1, int(depth1)))
        depth2 = min(4, max(1, int(depth2)))
    except:
        depth1 = depth2 = 2

    print("\nStart! Move formats: 'r c', 'A1', '1A', 'q' = quit.\n")
    print_board(b, a_pos, b_pos)

    cur = P1
    turn = 1

    # --- Main game loop ---
    while True:
        my_pos, opp_pos = (a_pos, b_pos) if cur == P1 else (b_pos, a_pos)
        legal = moves(b, my_pos, opp_pos)

        # No moves → current player loses
        if not legal:
            winner = P2 if cur == P1 else P1
            winner_icon = RED if winner == P2 else BLUE
            loser_icon = BLUE if cur == P1 else RED
            print(f"\n{loser_icon} cannot move — {winner_icon} wins!\n")

            final_board = [row[:] for row in b]
            if a_pos: final_board[a_pos[0]][a_pos[1]] = BLOCK
            if b_pos: final_board[b_pos[0]][b_pos[1]] = BLOCK
            print_board(final_board, None, None)
            break

        # Player move (human or AI)
        if mode == 'h' or (mode == 'a' and cur == P1):
            print(f"Player {BLUE if cur == P1 else RED}, available moves: " +
                  ", ".join(f"{chr(c+65)}{r+1}" for r, c in legal))
            mv = interactive_move_input(f"Move {BLUE if cur == P1 else RED}: ", R, C)
            if mv is None:
                print("Game ended.")
                break
            if mv not in legal:
                print("Invalid move, try again.")
                continue
        else:
            # AI move
            depth = depth1 if cur == P1 else depth2
            print(f"Turn {turn}: AI ({BLUE if cur == P1 else RED}) thinking (depth={depth})...")
            t0 = time.time()
            mv = ai_move(b, my_pos, opp_pos, depth)
            print(f"AI ({BLUE if cur == P1 else RED}) chooses {chr(mv[1]+65)}{mv[0]+1} in {time.time()-t0:.2f}s")

        # Apply move and update positions
        b, newp = apply_move(b, my_pos, mv)
        if cur == P1:
            a_pos = newp
        else:
            b_pos = newp

        print_board(b, a_pos, b_pos)

        if mode == 'aa':
            time.sleep(0.5)  # Small delay for readability in AI vs AI

        # Switch turn
        cur = P2 if cur == P1 else P1
        turn += 1


# --- Entry point ---
if __name__ == "__main__":
    play()
