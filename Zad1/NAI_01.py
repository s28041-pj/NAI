#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Isolation

Rules:
See the README.md file in this repository.
https://github.com/s28041-pj/NAI/blob/main/Zad1/readme.md

Authors:
- Mikołaj Gurgul, Łukasz Aleksandrowicz

To run the game you need to do below steps:

1. Install Python
2. Clone this repository
3. Run the game
    - python NAI_01.py
4. Enter row and cell number
5. Enter the max depth AI
6. Enjoy the game

The AI will respond automatically.
"""

from typing import List, Tuple, Optional
import math
import time

Board = List[List[str]]
Pos = Tuple[int, int]

EMPTY = '·'
BLOCK = '#'
P1 = 'A'
P2 = 'B'

BLUE = "🔵"
RED = "🔴"

DIRS = [(-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)]

def make_board(r: int, c: int) -> Board:
    return [[EMPTY for _ in range(c)] for _ in range(r)]

def in_bounds(b: Board, p: Pos) -> bool:
    r, c = p
    return 0 <= r < len(b) and 0 <= c < len(b[0])

def moves(b: Board, p: Pos, opp: Optional[Pos]) -> List[Pos]:
    if p is None:
        return []
    out = []
    for dr, dc in DIRS:
        np = (p[0] + dr, p[1] + dc)
        if in_bounds(b, np) and b[np[0]][np[1]] == EMPTY and np != opp:
            out.append(np)
    return out

def apply_move(b: Board, from_p: Pos, to_p: Pos) -> Tuple[Board, Pos]:
    nb = [row[:] for row in b]
    if from_p is not None:
        nb[from_p[0]][from_p[1]] = BLOCK
    return nb, to_p

def print_board(b: Board, a: Optional[Pos], bb: Optional[Pos]):
    cols = len(b[0])
    print("\n   " + "  ".join(chr(i + 65) for i in range(cols)))
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
    return len(moves(b, my, opp)) - len(moves(b, opp, my))

def negamax(b: Board, my: Pos, opp: Pos, depth: int) -> Tuple[int, Optional[Pos]]:
    legal = moves(b, my, opp)
    if not legal:
        return -9999, None
    if depth == 0:
        return heuristic(b, my, opp), None

    best_v = -math.inf
    best_m = None

    for m in legal:
        nb, new_my = apply_move(b, my, m)
        val, _ = negamax(nb, opp, new_my, depth - 1)
        val = -val
        if val > best_v:
            best_v = val
            best_m = m
    return int(best_v), best_m

def ai_move(b: Board, my: Pos, opp: Pos, depth: int = 2) -> Optional[Pos]:
    _, mv = negamax(b, my, opp, depth)
    return mv

def parse_token_pair(s: str, rows: int, cols: int) -> Optional[Pos]:
    s = s.strip().lower().replace(',', ' ').replace(';', ' ')
    parts = s.split()

    if len(parts) == 2:
        try:
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            if 0 <= r < rows and 0 <= c < cols:
                return (r, c)
        except Exception:
            return None

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
    s = input(prompt).strip()
    if s.lower() == 'q':
        return None

    pos = parse_token_pair(s, rows, cols)
    if pos is not None:
        return pos
    return None

def play():
    print("***Isolation***")
    rows = input("Wiersze (max 6, domyślnie 4): ").strip() or "4"
    cols = input("Kolumny (max 6, domyślnie 4): ").strip() or "4"
    try:
        R, C = min(6, max(2, int(rows))), min(6, max(2, int(cols)))
    except:
        R, C = 4, 4

    b = make_board(R, C)
    a_pos = (0, 0)
    b_pos = (R - 1, C - 1)

    mode = input("Tryb: human-human (h), human-AI (a), AI-AI (aa)? [a]: ").strip().lower() or 'a'

    depth1 = input("Głębokość AI (max 4, domyślnie 2): ").strip() or "2"
    depth2 = input("Głębokość AI (dla drugiego, jeśli AI-AI): ").strip() or depth1
    try:
        depth1 = min(4, max(1, int(depth1)))
        depth2 = min(4, max(1, int(depth2)))
    except:
        depth1 = depth2 = 2

    print("\nStart! Formaty ruchu: 'r c', 'A1', '1A', 'q' = wyjście.\n")
    print_board(b, a_pos, b_pos)

    cur = P1
    turn = 1

    while True:
        my_pos, opp_pos = (a_pos, b_pos) if cur == P1 else (b_pos, a_pos)
        legal = moves(b, my_pos, opp_pos)

        if not legal:
            winner = P2 if cur == P1 else P1
            winner_icon = RED if winner == P2 else BLUE
            loser_icon = BLUE if cur == P1 else RED
            print(f"\n{loser_icon} nie może się ruszyć — wygrywa {winner_icon}!\n")

            final_board = [row[:] for row in b]
            if a_pos: final_board[a_pos[0]][a_pos[1]] = BLOCK
            if b_pos: final_board[b_pos[0]][b_pos[1]] = BLOCK
            print_board(final_board, None, None)
            break

        if mode == 'h' or (mode == 'a' and cur == P1):
            print(f"Gracz {BLUE if cur == P1 else RED}, możliwe ruchy: " +
                  ", ".join(f"{chr(c+65)}{r+1}" for r, c in legal))
            mv = interactive_move_input(f"Ruch {BLUE if cur == P1 else RED}: ", R, C)
            if mv is None:
                print("Zakończono grę.")
                break
            if mv not in legal:
                print("Nielegalny ruch, spróbuj ponownie.")
                continue
        else:
            depth = depth1 if cur == P1 else depth2
            print(f"Tura {turn}: AI ({BLUE if cur == P1 else RED}) myśli (głębokość={depth})...")
            t0 = time.time()
            mv = ai_move(b, my_pos, opp_pos, depth)
            print(f"AI ({BLUE if cur == P1 else RED}) wybiera {chr(mv[1]+65)}{mv[0]+1} w {time.time()-t0:.2f}s")

        b, newp = apply_move(b, my_pos, mv)
        if cur == P1:
            a_pos = newp
        else:
            b_pos = newp

        print_board(b, a_pos, b_pos)

        if mode == 'aa':
            time.sleep(0.5)

        cur = P2 if cur == P1 else P1
        turn += 1

if __name__ == "__main__":
    play()
