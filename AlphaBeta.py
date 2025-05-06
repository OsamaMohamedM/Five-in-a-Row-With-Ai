import numpy as np
import hashlib
class Gomoku:
    def __init__(self, size=15):
        self.size = size                                # Set board dimension
        self.board = np.zeros((size, size), dtype=int)  # Board state: 0=empty, 1=AI/B, 2=human/W
        self.current_player = 1                         # Start with AI (1 for B), human plays as W (2)
        self.moves = []                                 # List to store move history for undo functionality
        self.max_depth = 4                              # Maximum depth for Alpha-Beta search 'to limit computation time'
        self.transposition_table = {}                   # Transposition table for caching evaluations

        """
            self.board = np.zeros((size, size), dtype=int)
            # Create a 2D NumPy array to represent the game board
            # np.zeros((size, size)) generates a size x size grid filled with zeros
            # (size, size) defines the shape: size rows and size columns (ex: 15x15 = 225 cells)
            # dtype=int ensures all elements are integers (0 for empty, 1 for AI/B, 2 for human/W)
        """

    def _get_board_hash(self):
        # Generate a hash of the current board state for the transposition table
        return hashlib.sha256(self.board.tobytes()).hexdigest()

    def _print_column_indices(self):
        print("  ", end=" ")
        for i in range(self.size):
            print(f"{i:2}", end=" ")
        print()

    def _print_row(self, row_idx):
        print(f"{row_idx:2}", end="  ")
        for j in range(self.size):
            if self.board[row_idx][j] == 0:
                print(".", end="  ")
            elif self.board[row_idx][j] == 1:
                print("B", end="  ")
            else:
                print("W", end="  ")
        print()

    def print_board(self):
        self._print_column_indices()
        for i in range(self.size):
            self._print_row(i)

    def is_valid_move(self, row, col):
        # Must be within board bounds and target an empty cell
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def make_move(self, row, col):
        # Place the current player's mark at (row, col) if valid
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player     # Set cell to player's mark (1 or 2)
            self.moves.append((row, col))                  # Record move for potential undo
            self.current_player = 3 - self.current_player  # Switch player: 1->2 or 2->1
            return True
        return False  # Return False if move is invalid

    def undo_move(self):
        # Remove the last move, used during Alpha-Beta search
        if self.moves:
            row, col = self.moves.pop()  # Remove last move from history
            self.board[row][col] = 0     # Clear the cell
            self.current_player = 3 - self.current_player  # Restore previous player
            return True
        return False  # Return False if no moves to undo

    def _count_sequence(self, i, j, dr, dc):
        # Count consecutive marks starting from (i, j) in direction (dr, dc)
        count = 1
        for k in range(1, 5):
            r = i + dr * k
            c = j + dc * k
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.board[i][j]:
                count += 1
            else:
                break
        return count

    def check_winner(self):
        # Check for a winner by finding five consecutive marks in a row
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal, anti-diagonal
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:  # Skip empty cells
                    continue
                for dr, dc in directions:
                    count = self._count_sequence(i, j, dr, dc)
                    if count >= 5:
                        return self.board[i][j]  # Return 1 (AI) or 2 (human) if five in a row
        return 0  # No winner found

    def is_board_full(self):
        # Check if the board is completely filled (no more moves possible)
        return len(self.moves) == self.size * self.size

    def _generate_neighboring_moves(self):
        # Generate neighboring moves around existing pieces
        moves = []
        for r, c in self.moves:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if self.is_valid_move(nr, nc) and (nr, nc) not in moves:
                        moves.append((nr, nc))
        return moves

    def check_winner_at(self, row, col):
        """
        Check if the last move at (row, col) created a winning condition (exactly five in a row).
        Returns: 0 (no winner), 1 (AI), or 2 (human).
        """
        if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row][col] == 0:
            return 0  # Invalid or empty position

        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal, anti-diagonal

        for dr, dc in directions:
            count = 1  # Current stone

            # Check in both directions (+ and -)
            for step in [1, -1]:
                for k in range(1, 5):  # Check up to 4 steps in each direction
                    r, c = row + dr * k * step, col + dc * k * step
                    if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                        count += 1
                    else:
                        break

            if count >= 5:
                return player  # Winner found

        return 0  # No winner
    def _find_opponent_threats(self):
        blocking_moves = []
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        opponent = 3 - self.current_player
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != opponent:
                    continue
                for dr, dc in directions:
                    count = self._count_sequence(i, j, dr, dc)
                    if count >= 4:  # Focus on four-in-a-row threats
                        # Check both ends of the sequence
                        start_r, start_c = i - dr, j - dc
                        end_r, end_c = i + dr * count, j + dc * count
                        if 0 <= start_r < self.size and 0 <= start_c < self.size and self.board[start_r][start_c] == 0:
                            blocking_moves.append((start_r, start_c))
                        if 0 <= end_r < self.size and 0 <= end_c < self.size and self.board[end_r][end_c] == 0:
                            blocking_moves.append((end_r, end_c))
        return list(set(blocking_moves))  # Remove duplicates

    def get_possible_moves(self):
        # First check all empty positions for immediate winning moves
        winning_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0 and self.check_winner_at(i, j):
                    winning_moves.append((i, j))
        if winning_moves:
            return winning_moves

        # Then check for opponent's threats that need blocking
        blocking_moves = self._find_opponent_threats()

        # Finally get neighboring moves if no urgent moves found
        neighboring_moves = self._generate_neighboring_moves()
        other_moves = [move for move in neighboring_moves if move not in blocking_moves]

        if not blocking_moves and not other_moves and not self.moves:
            center = self.size // 2
            other_moves.append((center, center))

        return blocking_moves + other_moves

    def _compute_heuristic_score(self):
        # First check if current player can win immediately
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0 and self.check_winner_at(i, j):
                    return 500000  # Highest possible score for winning move

        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    continue
                player = self.board[i][j]
                multiplier = 1 if player == 1 else -1
                for dr, dc in directions:
                    count = self._count_sequence(i, j, dr, dc)
                    if count < 2:
                        continue
                    # Check if sequence is open-ended
                    is_open_start = False
                    is_open_end = False
                    start_r= i - dr
                    start_c =j - dc
                    end_r, end_c = i + dr * count, j + dc * count
                    if 0 <= start_r < self.size and 0 <= start_c < self.size and self.board[start_r][start_c] == 0:
                        is_open_start = True
                    if 0 <= end_r < self.size and 0 <= end_c < self.size and self.board[end_r][end_c] == 0:
                        is_open_end = True
                    # Assign scores based on sequence length and openness
                    if count == 4:
                        if is_open_start and is_open_end:
                            score += multiplier * 100000  # Open four
                        elif is_open_start or is_open_end:
                            score += multiplier * 5000  # Half-open four
                        else:
                            score += multiplier * 1000  # Closed four
                        if player == 2:  # Opponent's four-in-a-row
                            score -= multiplier * 500000  # Increased penalty to force blocking
                    elif count == 3:
                        if is_open_start and is_open_end:
                            score += multiplier * 1000  # Open three
                        elif is_open_start or is_open_end:
                            score += multiplier * 500  # Half-open three
                        else:
                            score += multiplier * 100  # Closed three
                    elif count == 2:
                        if is_open_start and is_open_end:
                            score += multiplier * 100  # Open two
                        elif is_open_start or is_open_end:
                            score += multiplier * 50  # Half-open two
                    elif count >= 5:
                        score += multiplier * 500000  # Winning move
        return score

    def evaluate(self):
        # Evaluate the board state for the Alpha-Beta algorithm
        winner = self.check_winner()
        if winner == 1:
            return 1000
        if winner == 2:
            return -1000
        return self._compute_heuristic_score()

    def _maximize(self, depth, alpha, beta):
        # Maximizing player (AI's move) logic for Alpha-Beta Pruning
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
        # Minimizing player (human's move) logic for Alpha-Beta Pruning
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
        # Implement Alpha-Beta Pruning to find the best move
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

def _handle_ai_turn(game):
    print("AI is evaluating options...")

    # First, scan for any immediate winning move
    for move in game.get_possible_moves():
        game.make_move(*move)
        if game.check_winner_at(*move) == 1:
            print(f"AI plays winning move at {move}")
            return True  # Move already made
        game.undo_move()

    # If no immediate win, proceed with alpha-beta pruning
    _, move = game.alpha_beta(game.max_depth, float('-inf'), float('inf'), True)
    if move:
        game.make_move(*move)
        print(f"AI places B at {move}")
        return True

    print("No valid moves for AI.")
    return False



def _handle_human_turn(game):
    try:
        row = int(input("Enter row: "))
        col = int(input("Enter column: "))
        valid_moves = game.get_possible_moves()
        if not game.moves:
            center = game.size // 2
            valid_moves = [(center, center)]
        if (row, col) not in valid_moves:
            print("Invalid move")
            return False
        game.make_move(row, col)
        return True
    except ValueError:
        print("Please enter valid numbers.")
        return False


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


def human_vs_ai(game):
    print("Human (W) vs AI (B) using Alpha-Beta Pruning")
    game.current_player = 1
    while True:
        game.print_board()
        if game.current_player == 1:
            if not _handle_ai_turn(game):
                break
        else:
            if not _handle_human_turn(game):
                continue
        if _check_game_end(game):
            break


def main():
    game = Gomoku(size=10)
    print("Gomoku Game Solver: Human (W) vs AI (B)")
    human_vs_ai(game)

if __name__ == "__main__":
    main()
