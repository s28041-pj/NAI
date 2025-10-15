Game rules:

Each player moves one square at a time in any direction (diagonal, horizontal, or vertical).

After moving, the square the player left becomes blocked (#).

The player who cannot make a move loses.

Description:
A minimal implementation of the **Isolation** game in Python, supporting human vs human, human vs AI, and AI vs AI modes.
The board has a maximum size of 6×6, and the AI uses the **negamax** algorithm with configurable depth.
Players move across the board, and the square they leave becomes blocked (`#`). The game ends when a player cannot make a move.

Features:
- Board with columns labeled A–F and rows labeled 1–6.
- Colored players: 🔵 and 🔴 (displayed in the terminal).
- Game modes:
  - human vs human
  - human vs AI
  - AI vs AI
- Maximum board size: 6×6
- Maximum AI depth: 4
- Terminal-friendly board display with proper alignment

Enter moves in the following formats:

A1 or 1A

r c (e.g., 2 3)

r,c or r;c

If you enter only one token, the program will ask for the missing part.

Enter q to quit the game.

Notes:

AI can play at different difficulty levels (max depth = 4).

AI vs AI allows observing automatic matches.

The board and player markers are colored in the terminal (if ANSI colors are supported).