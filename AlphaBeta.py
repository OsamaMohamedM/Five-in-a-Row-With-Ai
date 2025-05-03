import numpy as np
class Gomoku:
    def __init__(self, size=15):
        self.size = size                                # Set board dimension
        self.board = np.zeros((size, size), dtype=int)  # Board state: 0=empty, 1=AI/B, 2=human/W
        self.current_player = 1                         # Start with AI (1 for B), human plays as W (2)
        self.moves = []                                 # List to store move history for undo functionality
        self.max_depth = 4                              # Maximum depth for Alpha-Beta search 'to limit computation time'

        """
            self.board = np.zeros((size, size), dtype=int)
            # Create a 2D NumPy array to represent the game board
            # np.zeros((size, size)) generates a size x size grid filled with zeros
            # (size, size) defines the shape: size rows and size columns (ex: 15x15 = 225 cells)
            # dtype=int ensures all elements are integers (0 for empty, 1 for AI/B, 2 for human/W)
        """

    def _print_column_indices(self):
        print("  ", end=" ")
        for i in range(self.size):
            print(f"{i:2}", end=" ")
        print()

    def _print_row(self, row_idx):
        print(f"{row_idx:2}", end="  ")
        for j in range(self.size):
            if self.board[row_idx][j] == 0:
                print(".", end="  ")  # Empty cell
            elif self.board[row_idx][j] == 1:
                print("B", end="  ")  # AI's mark
            else:
                print("W", end="  ")  # Human's mark
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

    def _check_if_winning_move(self, row, col):
        # Simulate a move to check if it wins for the current player
        self.board[row][col] = self.current_player
        is_winner = self.check_winner() == self.current_player
        self.board[row][col] = 0  # Remove the simulation
        return is_winner

    def get_possible_moves(self):
        # Generate possible moves, prioritizing winning moves
        winning_moves = []
        other_moves = []
        neighboring_moves = self._generate_neighboring_moves()
        for move in neighboring_moves:
            nr, nc = move
            if self._check_if_winning_move(nr, nc):
                winning_moves.append((nr, nc))
            else:
                other_moves.append((nr, nc))
        if not winning_moves and not other_moves and not self.moves:
            center = self.size // 2
            other_moves.append((center, center))
        return winning_moves + other_moves

    def _compute_heuristic_score(self):
        # Compute heuristic score based on sequences of 3 or more pieces
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    continue
                for dr, dc in directions:
                    count = self._count_sequence(i, j, dr, dc)
                    if count >= 3:
                        score += 10 * count if self.board[i][j] == 1 else -10 * count
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
        if depth == 0 or self.check_winner() != 0 or self.is_board_full():
            return self.evaluate(), None

        if maximizing_player:
            return self._maximize(depth, alpha, beta)
        else:
            return self._minimize(depth, alpha, beta)


def _handle_ai_turn(game):
    # Handle AI's turn using Alpha-Beta Pruning
    print("AI is evaluating options...")
    _, move = game.alpha_beta(game.max_depth, float('-inf'), float('inf'), True)
    if move:
        game.make_move(*move)
        print(f"AI places B at {move}")
        return True
    print("No valid moves for AI.")
    return False


def _handle_human_turn(game):
    # Handle human's turn by getting input, restricting to neighboring moves
    try:
        row = int(input("Enter row: "))
        col = int(input("Enter column: "))

        # Get valid neighboring moves
        valid_moves = game._generate_neighboring_moves()

        # If the board is empty, allow the center move
        if not game.moves:
            center = game.size // 2
            valid_moves = [(center, center)]

        # Check if the move is valid and in the list of neighboring moves
        if (row, col) not in valid_moves:
            print("Invalid move")
            return False

        game.make_move(row, col)

        return True
    except ValueError:
        print("Please enter valid numbers.")
        return False

def _check_game_end(game):
    # Check if the game has ended (winner or draw)
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
    game.current_player = 1  # AI starts as B
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
    game = Gomoku(size=15)
    print("Gomoku Game Solver: Human (W) vs AI (B)")
    human_vs_ai(game)

if __name__ == "__main__":
    main()

