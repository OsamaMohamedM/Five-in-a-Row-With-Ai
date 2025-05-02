# Five-in-a-Row-With-Ai

# Five in a Row AI Solver 🎯

A Python implementation of a  Five in a Row game solver, featuring Human vs AI and AI vs AI gameplay using classic game tree search algorithms: **Minimax** and **Alpha-Beta Pruning**.

---

## 🕹️ Game Overview

**Five in a Row With Ai** is a strategic board game where two players alternate placing marks (typically X and O) on a grid. The goal is to align **five consecutive marks** horizontally, vertically, or diagonally.

This project supports two game modes:
- **Human vs AI** (using the Minimax algorithm)
- **AI vs AI** (Minimax vs Alpha-Beta Pruning)

---

## 🚀 Features

- Configurable board sizes (e.g., 15x15, 19x19)
- Minimax and Alpha-Beta pruning AI agents
- Depth-limited search to optimize performance
- Interactive command-line interface for human moves
- Option to display the board after each move
- Clear code structure, modular and extendable

---

## 🗂️ Project Structure




---

## 🧠 Algorithms

### Minimax
A decision-making algorithm used in turn-based games. Explores all possible moves to a certain depth and chooses the optimal one assuming the opponent plays optimally.

### Alpha-Beta Pruning
An optimization over Minimax that skips branches that won’t affect the final decision, drastically reducing the number of nodes evaluated.

---

## 💻 Installation

Make sure you have **Python 3.7+** installed.

```bash
# Clone the repository
git clone https://github.com/yourusername/Five-in-a-Row-With-Ai.git
cd Game
# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
## Usage
python main.py

## Input & Output
Input:
Interactive player moves (row, column)
Optional: Load a custom board state

Output:
AI-selected move coordinates

Updated board display in console (or GUI if integrated)
## Technologies Used
Python 3.x
Standard Python libraries

