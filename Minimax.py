"""
Gomoku (Five-in-a-Row) Game Implementation

This program implements a Gomoku game where players take turns placing stones on a 15x15 board,
aiming to be the first to get five stones in a row (horizontally, vertically, or diagonally).

Key Components:
1. Game Board: 15x15 grid initialized with empty spaces ('.')
2. Players:
   - 'B' for Black (typically goes first)
   - 'W' for White
3. Game Modes:
   - Human vs AI (using Minimax algorithm)
   - AI vs AI (for demonstration)
4. AI Implementation:
   - Uses Minimax algorithm with depth-limited search
   - Board evaluation based on pattern recognition
   - Move generation optimized with candidate moves

Core Functions:
- initialize_board(): Creates empty game board
- is_winner(): Checks if a player has five in a row
- is_draw(): Checks if board is full
- evaluate_board(): Scores board positions for AI
- minimax(): AI decision-making algorithm
- get_human_move(): Handles human player input

Game Flow:
1. Players alternate turns placing stones
2. After each move, checks for win/draw conditions
3. AI uses Minimax to determine best move
4. Game ends when a player wins or board is full

The implementation includes optimizations:
- Candidate move generation to reduce search space
- Memoization to cache board evaluations
- Pattern-based scoring for efficient evaluation
"""

import copy

# Game Constants
BOARD_SIZE = 10  # Standard Gomoku board size (15x15 grid)
DEPTH_SIZE = 3  # Depth for AI search (higher = stronger but slower)

# Direction vectors for checking lines (8 directions: horizontal, vertical, diagonal)
dx = [1, 0, -1, 0, -1, -1, 1, 1]  # Column directions (x-axis)
dy = [0, -1, 0, 1, -1, 1, -1, 1]  # Row directions (y-axis)


def is_valid(x, y):
    """Check if coordinates (x,y) are within board boundaries.
    Args:
        x (int): Row index
        y (int): Column index
    Returns:
        bool: True if valid position, False otherwise
    """
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def is_winner(board, player):
    """Check if specified player has won (5 stones in a row).
    Args:
        board (list): 2D list representing the game board
        player (str): 'B' or 'W' representing the player to check
    Returns:
        bool: True if player has won, False otherwise
    """
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] != player:
                continue  # Skip positions not occupied by player

            # Check all 8 possible directions for 5-in-a-row
            for i in range(8):
                count = 1  # Start counting with current stone

                # Check forward direction
                for j in range(1, 5):
                    nx, ny = x + dx[i] * j, y + dy[i] * j
                    if is_valid(nx, ny) and board[nx][ny] == player:
                        count += 1
                    else:
                        break

                # Check backward direction
                for j in range(1, 5):
                    nx, ny = x - dx[i] * j, y - dy[i] * j
                    if is_valid(nx, ny) and board[nx][ny] == player:
                        count += 1
                    else:
                        break

                if count >= 5:  # Winning condition met
                    return True
    return False


def is_draw(board):
    """Check if the game is a draw (board full with no winner).
    Args:
        board (list): 2D list representing the game board
    Returns:
        bool: True if game is draw, False otherwise
    """
    # Check if any empty space remains
    for row in board:
        if '.' in row:  # Empty space exists
            return False
    # Board is full and no winner
    return not (is_winner(board, 'B') or is_winner(board, 'W'))


def initialize_board():
    """Create and return a new empty game board.
    Returns:
        list: 2D list of size BOARD_SIZE x BOARD_SIZE filled with '.'
    """
    return [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def print_column_indices():
    print("  ", end=" ")
    for i in range(BOARD_SIZE):
        print(f"{i:2}", end=" ")
    print()


def print_row(board, row_idx):
    print(f"{row_idx:2}", end="  ")
    for j in range(BOARD_SIZE):
        print(board[row_idx][j], end="  ")
    print()


def print_board(board):
    print_column_indices()
    for i in range(BOARD_SIZE):
        print_row(board , i)  # Print each row with index


# Print each row as space-separated characters


def get_candidate_moves(board, initial_radius=1):
    """Generate potential moves near existing stones for efficiency.
    Args:
        board (list): Current game board
        initial_radius (int): Search radius around existing stones
    Returns:
        list: List of valid (x,y) move coordinates
    """
    candidates = set()

    # First move must be center (standard Gomoku rule)
    if all(cell == '.' for row in board for cell in row):
        return [(5, 5)]

    # Phase 1: Check within radius of existing stones
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] != '.':
                for d in range(8):  # Check all 8 directions
                    for r in range(1, initial_radius + 1):
                        nx, ny = x + dx[d] * r, y + dy[d] * r
                        if is_valid(nx, ny) and board[nx][ny] == '.':
                            candidates.add((nx, ny))

    # Phase 2: If no candidates found, search entire board
    if not candidates:
        candidates = {(x, y) for x in range(BOARD_SIZE)
                      for y in range(BOARD_SIZE) if board[x][y] == '.'}

    return list(candidates)


def get_human_move(board):
    """Get and validate human player's move.
    Args:
        board (list): Current game board
    Returns:
        tuple: (x,y) coordinates of valid move
    """
    valid_moves = get_candidate_moves(board)
    while True:
        try:
            print("\nValid moves: ", valid_moves)
            x, y = map(int, input("Enter your move (row and column): ").split())
            if (x, y) in valid_moves:
                return x, y
            print("Invalid move! Try again.")
        except ValueError:
            print("Invalid input! Enter two numbers (e.g., '7 7').")
        except Exception as e:
            print(f"Error: {e}")


def make_move(board, x, y, player):
    """Place a stone on the board if position is empty.
    Args:
        board (list): Current game board
        x (int): Row index
        y (int): Column index
        player (str): 'B' or 'W'
    Returns:
        bool: True if move was valid, False otherwise
    """
    if board[x][y] == '.':
        board[x][y] = player
        return True
    return False


def evaluate_board(board, player):
    opponent = 'W' if player == 'B' else 'B'
    score = 0

    # Probability-inspired pattern weights
    weights = {
        (5, 2): 100000,  # Open 5 (instant win)
        (4, 2): 10000,   # Open 4 (almost guaranteed win)
        (4, 1): 2000,    # Blocked 4
        (3, 2): 1500,    # Open 3
        (3, 1): 300,     # Blocked 3
        (2, 2): 100,     # Open 2
        (2, 1): 30,      # Blocked 2
        (1, 2): 10       # Single stone with room to grow
    }

    def pattern_score(x, y, dx, dy, player):
        count = 0
        open_ends = 0
        # Forward
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if not is_valid(nx, ny):
                break
            if board[nx][ny] == player:
                count += 1
            elif board[nx][ny] == '.':
                open_ends += 1
                break
            else:
                break
        # Backward
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if not is_valid(nx, ny):
                break
            if board[nx][ny] == player:
                count += 1
            elif board[nx][ny] == '.':
                open_ends += 1
                break
            else:
                break
        return weights.get((count, open_ends), 0)

    # Evaluate board
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == '.':
                continue
            for d in range(8):
                if board[x][y] == player:
                    score += pattern_score(x, y, dx[d], dy[d], player)
                elif board[x][y] == opponent:
                    score -= pattern_score(x, y, dx[d], dy[d], opponent)

    # Center control bonus
    center = BOARD_SIZE // 2
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == player:
                distance = abs(center - x) + abs(center - y)
                score += max(0, 10 - distance)  # Bonus for center proximity

    return score

# Memoization cache to store previously evaluated board states
memo = {}

def board_to_key(board):
    """Convert board state to a hashable key for memoization.
    Args:
        board (list): Current game board
    Returns:
        tuple: Immutable representation of board state
    """
    return tuple("".join(row) for row in board)

def minimax(board, depth, is_maximizing, player):
    """Minimax algorithm with memoization for AI move decision.
    Args:
        board (list): Current game board
        depth (int): Current search depth
        is_maximizing (bool): True if maximizing player's turn
        player (str): Current player ('B' or 'W')
    Returns:
        tuple: (evaluation_score, best_move)
    """
    key = (board_to_key(board), depth, is_maximizing)
    if key in memo:
        return memo[key]

    opponent = 'W' if player == 'B' else 'B'

    # Base case: terminal state or depth limit reached
    if depth == 0 or is_winner(board, player) or is_winner(board, opponent) or is_draw(board):
        score = evaluate_board(board, player)
        memo[key] = (score, None)
        return score, None

    moves = get_candidate_moves(board)
    if not moves:
        moves = [(BOARD_SIZE//2, BOARD_SIZE//2)]  # Default to center if no moves found (shouldn't happen)

    best_move = None

    if is_maximizing:
        max_eval = float('-inf')
        for x, y in moves:
            new_board = copy.deepcopy(board)
            new_board[x][y] = player
            eval_score, _ = minimax(new_board, depth - 1, False, player)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = (x, y)
        memo[key] = (max_eval, best_move)
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for x, y in moves:
            new_board = copy.deepcopy(board)
            new_board[x][y] = opponent
            eval_score, _ = minimax(new_board, depth - 1, True, player)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = (x, y)
        memo[key] = (min_eval, best_move)
        return min_eval, best_move


def minimax_move(board, depth, player):
    """Wrapper function to get best move using minimax.
    Args:
        board (list): Current game board
        depth (int): Search depth
        player (str): 'B' or 'W' representing AI player
    Returns:
        tuple: (x,y) coordinates of best move
    """
    _, move = minimax(board, depth, True, player)
    return move


def display_menu():
    """Display game menu and get user choice.
    Returns:
        str: User's menu choice
    """
    print("\n===== Gomoku Menu =====")
    print("1. Human vs AI (Minimax)")
    print("2. AI vs AI (Minimax vs Alpha-Beta)")
    print("3. Exit")
    return input("Choose an option (1-3): ").strip()


def play_game(human_vs_ai=True):
    """Main game loop for different game modes.
    Args:
        human_vs_ai (bool): True for human vs AI, False for AI vs AI
    """
    board = initialize_board()
    current_turn = 'B'  # Black always starts first

    # Set player roles
    if human_vs_ai:
        human_player = input("Choose your color [B/W]: ").upper()
        while human_player not in ('B', 'W'):
            human_player = input("Invalid! Choose 'B' or 'W': ").upper()
        ai_player = 'W' if human_player == 'B' else 'B'
    else:
        human_player = None
        ai_player = 'B'

    print_board(board)

    # Game loop
    while True:
        if human_vs_ai and current_turn == human_player:
            x, y = get_human_move(board)  # Human move
        else:
            print(f"\nAI ({current_turn}) is thinking...")
            x, y = minimax_move(board, DEPTH_SIZE, current_turn)  # AI move
            print(f"AI plays at ({x}, {y})")

        make_move(board, x, y, current_turn)
        print_board(board)

        # Check game state
        if is_winner(board, current_turn):
            print(f"{current_turn} wins!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        current_turn = 'W' if current_turn == 'B' else 'B'  # Switch turns


def main():
    """Main program entry point."""
    while True:
        choice = display_menu()

        if choice == "1":
            print("\n--- Human vs AI (Minimax) ---")
            play_game(human_vs_ai=True)
        elif choice == "2":
            print("\n--- AI vs AI (Minimax vs Alpha-Beta) ---")
            play_game(human_vs_ai=False)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Try again.")


main()
