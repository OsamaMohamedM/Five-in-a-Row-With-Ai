import numpy as np
import hashlib

class Gomoku:
    def __init__(self, size=15):
        """Initialize a Gomoku game with a square board of given size."""
        self.size = size                                      # Board dimension (default 15x15)
        self.board = np.zeros((size, size), dtype=int)        # Board state: 0=empty, 1=AI/Black, 2=Human/White or another AI in mode (AI VS AI)
        self.current_player = 1                               # AI (Black) starts; Human is White
        self.moves = []                                       # Store move history for undo
        self.max_depth = 4                                    # Depth limit for AI search
        self.transposition_table = {}                         # Cache for board evaluations
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]   # Horizontal, vertical, diagonal, anti-diagonal

        """
            self.board = np.zeros((size, size), dtype=int)
            # Create a 2D NumPy array to represent the game board
            # np.zeros((size, size)) generates a size(rows) x size(columns) grid filled with zeros
            # dtype=int ensures all elements are integers (0 for empty, 1 for AI/B, 2 for human/W)
        """

    def _get_board_hash(self):
        """Generate a SHA-256 hash of the current board state for caching."""
        return hashlib.sha256(self.board.tobytes()).hexdigest()

    def print_board(self):
        """Display the current board state with indices."""
        # Print column indices
        print("   " + " ".join(f"{i:2}" for i in range(self.size)))

        # Print each row with row index and pieces
        for i in range(self.size):
            row_str = f"{i:2} "
            for j in range(self.size):
                cell = self.board[i][j]
                row_str += " . " if cell == 0 else " B " if cell == 1 else " W "
            print(row_str)

    def is_valid_move(self, row, col):
        """Check if a move is valid (within bounds and cell is empty)."""
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def make_move(self, row, col):
        """Place current player's piece at (row, col) if valid and switch player."""
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.moves.append((row, col))                  # Append the move to the moves list for potential undo functionality
            self.current_player = 3 - self.current_player  # Switch: 1->2, 2->1
            return True                                    # indicating that the move was successfully made
        return False                                       # if the move is invalid (either out of bounds or targeting a non-empty cell)

    def undo_move(self):
        """Undo the last move and switch back to previous player."""
        if not self.moves:
            return False
        row, col = self.moves.pop()
        self.board[row][col] = 0
        self.current_player = 3 - self.current_player
        return True

    def _count_sequence(self, i, j, dr, dc):
        """Count consecutive pieces from (i, j) in direction (dr, dc)."""
        count = 1
        for step in range(1, 5):     # Check up to 4 more cells in the specified direction (dr, dc)
            r = i + dr * step
            c = j + dc * step
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.board[i][j]:
                count += 1
            else:
                break
        return count

    def check_winner(self):
        """Check for a winner by finding five consecutive pieces."""
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:  # Skip empty cells
                    continue
                for dr, dc in self.directions:
                    if self._count_sequence(i, j, dr, dc) >= 5:
                        return self.board[i][j]  # Return player ID (1 for Black, 2 for White)
        return 0  # No winner

    def check_winner_at(self, row, col):
        """Check if a move at (row,col) creates a win."""
        if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row][col] == 0:
            return 0
        player = self.board[row][col]
        for dr, dc in self.directions:
            count = 1
            # Check in both directions (+ and -)
            for step in [1, -1]:
                for k in range(1, 5):
                    r = row + dr * k * step
                    c = col + dc * k * step
                    if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                        count += 1
                    else:
                        break
            if count >= 5:
                return player
        return 0

    def is_board_full(self):
        """Check if the board is full (no empty cells)."""
        return len(self.moves) == self.size * self.size

    def _generate_neighboring_moves(self):
        """Generate valid moves adjacent to existing pieces."""
        neighboring_moves = []
        for r, c in self.moves:
            for dr in [-1, 0, 1]:               # Check all row offsets: up (-1), same (0), down (1)
                for dc in [-1, 0, 1]:           # Check all column offsets: left (-1), same (0), right (1)
                    nr = r + dr
                    nc = c + dc
                    if self.is_valid_move(nr, nc) and (nr,nc) not in neighboring_moves:
                        neighboring_moves.append((nr, nc))
        return neighboring_moves

    def _find_opponent_threats(self):
        """Identify moves to block opponent's four-in-a-row threats."""
        blocking_moves = []
        opponent = 3 - self.current_player        # Determine opponent ID
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != opponent:  # Skip non-opponent pieces
                    continue
                for dr, dc in self.directions:
                    count = self._count_sequence(i, j, dr,dc)
                    if count >= 4:
                        # Check both ends of the sequence for empty cells where a blocking move can be made.
                        start_r = i - dr
                        start_c = j - dc
                        end_r = i + dr * count
                        end_c = j + dc * count
                        if self.is_valid_move(start_r,start_c):
                            blocking_moves.append((start_r, start_c))
                        if self.is_valid_move(end_r,end_c):
                            blocking_moves.append((end_r, end_c))
        return list(set(blocking_moves)) # Remove duplicates

    def get_possible_moves(self):
        # First check all empty positions for immediate winning moves
        winning_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0 and self.check_winner_at(i, j):
                    winning_moves.append((i, j))
        if winning_moves:
            return winning_moves

        # Then check for opponent's threats
        blocking_moves = self._find_opponent_threats()

        # Finally get neighboring moves
        neighboring_moves = self._generate_neighboring_moves()
        other_moves = [move for move in neighboring_moves if move not in blocking_moves]

        # Fallback to center if board is empty
        if not blocking_moves and not other_moves and not self.moves:
            center = self.size // 2
            other_moves.append((center, center))

        return blocking_moves + other_moves

    def evaluate(self):
        """Evaluate the board state for the AI."""
        winner = self.check_winner()
        if winner == 1:
            return float('inf')
        if winner == 2:
            return float('-inf')

        score = 0
        patterns = {
            (4, True): 100000,  # Open four
            (4, False): 5000,   # Half-open four
            (3, True): 1000,    # Open three
            (3, False): 500,    # Half-open three
            (2, True): 100,     # Open two
            (2, False): 50      # Half-open two
        }

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    continue
                player = self.board[i][j]
                multiplier = 1 if player == 1 else -1
                for dr, dc in self.directions:
                    count = self._count_sequence(i, j, dr, dc)
                    if count < 2:
                        continue
                    # Check if sequence is open-ended
                    open_start = self.is_valid_move(i - dr, j - dc)
                    open_end = self.is_valid_move(i + dr * count, j + dc * count)
                    is_open = open_start and open_end
                    key = (count, is_open) if count in (2, 3, 4) else (count, False)
                    if key in patterns:
                        score += multiplier * patterns[key]
        return score

    def _maximize(self, depth, alpha, beta):
        """Maximizing player (AI) logic for Alpha-Beta pruning."""
        max_eval = float('-inf')
        best_move = None
        for move in self.get_possible_moves():
            self.make_move(*move)
            eval_score, _ = self.alpha_beta(depth - 1, alpha, beta, False)
            self.undo_move()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move

    def _minimize(self, depth, alpha, beta):
        """ Minimizing player (human's move) logic for Alpha-Beta Pruning"""
        min_eval = float('inf')
        best_move = None
        for move in self.get_possible_moves():
            self.make_move(*move)
            eval_score, _ = self.alpha_beta(depth - 1, alpha, beta, True)
            self.undo_move()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        """Implement Alpha-Beta Pruning to find the best move"""
        board_hash = self._get_board_hash()
        tt_entry = self.transposition_table.get((board_hash, depth, maximizing_player))
        if tt_entry is not None:
            return tt_entry  # Return cached result if available

        if depth == 0 or self.check_winner() != 0 or self.is_board_full():
            eval_score = self.evaluate()
            self.transposition_table[(board_hash, depth, maximizing_player)] = (eval_score, None)
            return eval_score, None

        if maximizing_player:
            result = self._maximize(depth, alpha, beta)
        else:
            result = self._minimize(depth, alpha, beta)
        self.transposition_table[(board_hash, depth, maximizing_player)] = result
        return result

    def minimax_maximize(self, depth):
        """Maximizing player (AI's move) logic for Minimax"""
        max_eval = float('-inf')
        best_move = None
        for move in self.get_possible_moves():
            self.make_move(*move)
            eval_score, _ = self.minimax(depth - 1, False)
            self.undo_move()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        return max_eval, best_move

    def minimax_minimize(self, depth):
        """Minimizing player (human's move) logic for Minimax"""
        min_eval = float('inf')
        best_move = None
        for move in self.get_possible_moves():
            self.make_move(*move)
            eval_score, _ = self.minimax(depth - 1, True)
            self.undo_move()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
        return min_eval, best_move

    def minimax(self, depth, maximizing_player):
        """Implement Minimax to find the best move."""
        board_hash = self._get_board_hash()
        tt_entry = self.transposition_table.get((board_hash, depth, maximizing_player))
        if tt_entry is not None:
            return tt_entry  # Return cached result if available

        if depth == 0 or self.check_winner() != 0 or self.is_board_full():
            eval_score = self.evaluate()
            self.transposition_table[(board_hash, depth, maximizing_player)] = (eval_score, None)
            return eval_score, None

        if maximizing_player:
            result = self.minimax_maximize(depth)
        else:
            result = self.minimax_minimize(depth)
        self.transposition_table[(board_hash, depth, maximizing_player)] = result
        return result

"""============================================================================================================================================="""

def _check_game_end(game):
    winner = game.check_winner()
    if winner:
        game.print_board()
        print(f"Player {'B' if winner == 1 else 'W'} wins!")
        return True
    if game.is_board_full():
        game.print_board()
        print("It's a draw!")
        return True
    return False

def human_vs_ai(game, use_alpha_beta=True):
    """Human vs AI game loop."""
    print(f"\nHuman (W) vs AI (B) using {'Alpha-Beta' if use_alpha_beta else 'Minimax'}")
    while True:
        game.print_board()

        if game.check_winner() or game.is_board_full():
            break

        if game.current_player == 1:  # AI's turn
            print("AI is thinking...")
            # Check for immediate winning move
            made_move = False
            for move in game.get_possible_moves():
                game.make_move(*move)
                if game.check_winner_at(*move) == 1:
                    print(f"AI plays winning move at {move}")
                    made_move = True
                    break
                game.undo_move()
            if not made_move:
                # No immediate win, use selected algorithm
                if use_alpha_beta:
                    _, move = game.alpha_beta(game.max_depth, float('-inf'), float('inf'), True)
                else:
                    _, move = game.minimax(game.max_depth, True)
                if move:
                    game.make_move(*move)
                    print(f"AI places B at {move}")
        else:  # Human's turn
            try:
                valid_move=game._generate_neighboring_moves()
                print("\nValid Moves",valid_move)
                row = int(input("Enter row: "))
                col = int(input("Enter column: "))
                if(row,col) not in valid_move:
                    print("Invalid move. Choose move from valid list")
                    continue
                game.make_move(row, col)
            except ValueError:
                print("Please enter numbers only.")

    # Game ended
    _check_game_end(game)


def ai_vs_ai(game):
    """AI vs AI game loop (Minimax vs Alpha-Beta)."""
    print("\nAI (Minimax as B) vs AI (Alpha-Beta as W)")
    game.current_player = 1  # Minimax starts first

    while True:
        game.print_board()
        current_player_name = 'Minimax (B)' if game.current_player == 1 else 'Alpha-Beta (W)'
        print(f"{current_player_name} is thinking...")

        # Check for immediate winning move
        made_move = False
        for move in game.get_possible_moves():
            game.make_move(*move)
            if game.check_winner_at(*move) == game.current_player:
                print(f"{current_player_name} plays winning move at {move}")
                made_move = True
                break
            game.undo_move()

        if not made_move:
            # Check for opponent's threats using _find_opponent_threats
            blocking_moves = game._find_opponent_threats()
            if blocking_moves:
                # Choose the first blocking move (or improve with evaluation if needed)
                move = blocking_moves[0]
                game.make_move(*move)
                print(f"{current_player_name} blocks opponent's winning move at {move}")
            else:
                # No immediate threats, use AI algorithm
                if game.current_player == 1:
                    _, move = game.minimax(game.max_depth, True)
                else:
                    _, move = game.alpha_beta(game.max_depth, float('-inf'), float('inf'), False)

                if move:
                    game.make_move(*move)
                    print(f"{current_player_name} places at {move}")
                else:
                    print("No valid moves for AI.")
                    break

        if _check_game_end(game):
            break
def display_menu():
    print("\n===== Gomoku Menu =====")
    print("1. Human vs AI (Minimax)")
    print("2. Human vs AI (Alpha-Beta)")
    print("3. AI vs AI (Minimax vs Alpha-Beta)")
    print("4. Exit")
    return input("Choose an option (1-4): ").strip()


def main():
    size = int(input("Enter board size (default 15): ") or 15)

    while True:
        choice = display_menu()

        if choice == "1":
            game = Gomoku(size=size)
            human_vs_ai(game, use_alpha_beta=False)
        elif choice == "2":
            game = Gomoku(size=size)
            human_vs_ai(game, use_alpha_beta=True)
        elif choice == "3":
            game = Gomoku(size=size)
            ai_vs_ai(game)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Try again.")

if __name__ == "__main__":
    main()
