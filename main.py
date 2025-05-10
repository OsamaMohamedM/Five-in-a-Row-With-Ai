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
        return hashlib.sha256(
            self.board.tobytes()).hexdigest()  # Convert the board state to bytes and compute its SHA-256 hash, then return the hexadecimal string.

    def _print_column_indices(self):
        print("  ", end=" ")  # Print two spaces as a header padding for alignment.
        for i in range(self.size):  # Iterate through the column indices from 0 to self.size-1
            print(f"{i:2}",
                  end=" ")  # Print each column index, ensuring it's formatted with a width of 2 for alignment.
        print()  # Print a newline after printing all column indices.

    def _print_row(self, row_idx):
        print(f"{row_idx:2}",
              end="  ")  # Print the row index (formatted with a width of 2) followed by two spaces for alignment.
        for j in range(self.size):  # Iterate through each column in the given row (from 0 to self.size-1).
            if self.board[row_idx][j] == 0:  # Check if the cell is empty (represented by 0).
                print(".", end="  ")  # Print a dot (.) to represent an empty cell.
            elif self.board[row_idx][j] == 1:  # Check if the cell contains a 'B' (black piece).
                print("B", end="  ")  # Print 'B' to represent a black piece.
            else:  # If the cell contains something other than 0 or 1, it must be a white piece.
                print("W", end="  ")  # Print 'W' to represent a white piece.
        print()  # Print a newline after printing all the cells of the row.

    def print_board(self):
        self._print_column_indices()  # Print the column indices at the top of the board.
        for i in range(self.size):  # Iterate through each row of the board.
            self._print_row(i)  # Print the current row using the _print_row method.

    def is_valid_move(self, row, col):
        # Must be within board bounds and target an empty cell
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][
            col] == 0  # Check if the row and column are within the valid board range and if the target cell is empty (0).

    def make_move(self, row, col):
        # Place the current player's mark at (row, col) if valid
        if self.is_valid_move(row, col):  # Check if the move is valid using the is_valid_move method.
            self.board[row][
                col] = self.current_player  # Set the selected cell to the current player's mark (either 1 or 2).
            self.moves.append((row, col))  # Append the move to the moves list for potential undo functionality.
            self.current_player = 3 - self.current_player  # Switch players: If current_player is 1, set to 2, and vice versa.
            return True  # Return True indicating that the move was successfully made.
        return False  # Return False if the move is invalid (either out of bounds or targeting a non-empty cell).

    def undo_move(self):
        # Remove the last move, used during Alpha-Beta search
        if self.moves:  # Check if there are any moves to undo (if the moves list is not empty).
            row, col = self.moves.pop()  # Remove the last move from the moves history (the most recent move).
            self.board[row][col] = 0  # Clear the cell by resetting it to 0 (empty).
            self.current_player = 3 - self.current_player  # Switch back to the previous player (1 <-> 2).
            return True  # Return True to indicate that the move was successfully undone.
        return False  # Return False if there are no moves to undo (moves list is empty).

    def _count_sequence(self, i, j, dr, dc):
        # Count consecutive marks starting from (i, j) in direction (dr, dc)
        count = 1  # Start with a count of 1 for the initial cell (i, j).
        for k in range(1, 5):  # Check up to 4 more cells in the specified direction (dr, dc).
            r = i + dr * k  # Calculate the row for the next cell in the sequence.
            c = j + dc * k  # Calculate the column for the next cell in the sequence.
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.board[i][j]:
                count += 1  # If the cell is within bounds and has the same mark as the starting cell, increment the count.
            else:
                break  # Stop counting if the mark does not match or the cell is out of bounds.
        return count  # Return the total count of consecutive marks in the specified direction.

    def check_winner(self):
        # Check for a winner by finding five consecutive marks in a row
        directions = [(0, 1), (1, 0), (1, 1),
                      (1, -1)]  # Define the four possible directions: horizontal, vertical, diagonal, anti-diagonal.
        for i in range(self.size):  # Iterate through each row of the board.
            for j in range(self.size):  # Iterate through each column of the board.
                if self.board[i][j] == 0:  # Skip empty cells.
                    continue
                for dr, dc in directions:  # Check each direction for a sequence of 5 consecutive marks.
                    count = self._count_sequence(i, j, dr,
                                                 dc)  # Count the consecutive marks starting from (i, j) in the direction (dr, dc).
                    if count >= 5:  # If there are 5 or more consecutive marks in this direction.
                        return self.board[i][
                            j]  # Return the player (1 for AI or 2 for human) who has 5 consecutive marks.
        return 0  # No winner found, return 0 indicating no winner.

    def is_board_full(self):
        # Check if the board is completely filled (no more moves possible)
        return len(
            self.moves) == self.size * self.size  # The board is full if the number of moves equals the total number of cells on the board.

    def _generate_neighboring_moves(self):
        # Generate neighboring moves around existing pieces
        moves = []  # Initialize an empty list to store valid neighboring moves.
        for r, c in self.moves:  # Iterate through each move that has been made (each piece's position).
            for dr in [-1, 0,
                       1]:  # Iterate through the possible row direction offsets: -1 (up), 0 (same row), 1 (down).
                for dc in [-1, 0,
                           1]:  # Iterate through the possible column direction offsets: -1 (left), 0 (same column), 1 (right).
                    nr, nc = r + dr, c + dc  # Calculate the new row (nr) and column (nc) based on the current piece position and the direction offsets.
                    if self.is_valid_move(nr, nc) and (nr,
                                                       nc) not in moves:  # Check if the new position is valid and hasn't been added to the moves list yet.
                        moves.append((nr, nc))  # If valid, add the new position (nr, nc) to the moves list.
        return moves  # Return the list of valid neighboring moves.

    def check_winner_at(self, row, col):
        """
        Check if the last move at (row, col) created a winning condition (exactly five in a row).
        Returns: 0 (no winner), 1 (AI), or 2 (human).
        """
        if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row][col] == 0:
            return 0  # If the position is invalid (out of bounds) or the cell is empty (0), return 0 indicating no winner.
        player = self.board[row][col]  # Determine which player (1 or 2) made the move.
        directions = [(0, 1), (1, 0), (1, 1), (1,-1)]  # Define the four directions to check for five consecutive marks: horizontal, vertical, diagonal, anti-diagonal.
        for dr, dc in directions:  # Loop through each direction.
            count = 1  # Start with the current stone as the first in the sequence.
            # Check in both directions (+ and -)
            for step in [1, -1]:  # Loop through both directions: +1 (positive direction) and -1 (negative direction).
                for k in range(1, 5):  # Check up to 4 steps in the current direction.
                    r, c = row + dr * k * step, col + dc * k * step  # Calculate the next position in the sequence.
                    if 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                        count += 1  # If the next cell matches the player's mark, increment the count.
                    else:
                        break  # If the cell is invalid or doesn't match the player's mark, stop checking this direction.
            if count >= 5:  # If 5 or more consecutive marks are found, a winner is declared.
                return player  # Return the player (1 or 2) as the winner.
        return 0  # If no sequence of 5 marks is found, return 0 indicating no winner.

    def _find_opponent_threats(self):
        blocking_moves = []  # Initialize an empty list to store potential moves to block the opponent's threats.
        directions = [(0, 1), (1, 0), (1, 1),
                      (1, -1)]  # Define the four possible directions: horizontal, vertical, diagonal, anti-diagonal.
        opponent = 3 - self.current_player  # Determine the opponent's player (if current player is 1, opponent is 2, and vice versa).
        for i in range(self.size):  # Iterate through each row of the board.
            for j in range(self.size):  # Iterate through each column of the board.
                if self.board[i][j] != opponent:  # Skip non-opponent pieces.
                    continue
                for dr, dc in directions:  # Check each direction for possible four-in-a-row threats.
                    count = self._count_sequence(i, j, dr,dc)  # Count the consecutive opponent marks starting from (i, j) in direction (dr, dc).
                    if count >= 4:  # If there are 4 consecutive opponent marks in this direction, it's a threat.
                        # Check both ends of the sequence for empty cells where a blocking move can be made.
                        start_r, start_c = i - dr, j - dc  # Calculate the position before the sequence (start of potential block).
                        end_r, end_c = i + dr * count, j + dc * count  # Calculate the position after the sequence (end of potential block).
                        # Check if the start position is within bounds and empty.
                        if 0 <= start_r < self.size and 0 <= start_c < self.size and self.board[start_r][start_c] == 0:
                            blocking_moves.append(
                                (start_r, start_c))  # Add to blocking moves if the start is valid and empty.
                        # Check if the end position is within bounds and empty.
                        if 0 <= end_r < self.size and 0 <= end_c < self.size and self.board[end_r][end_c] == 0:
                            blocking_moves.append(
                                (end_r, end_c))  # Add to blocking moves if the end is valid and empty.
        return list(set(blocking_moves))  # Remove duplicates from the blocking moves list and return it.

    def get_possible_moves(self):
        # First check all empty positions for immediate winning moves
        winning_moves = []  # Initialize an empty list to store winning moves.
        for i in range(self.size):  # Iterate through each row.
            for j in range(self.size):  # Iterate through each column.
                if self.board[i][j] == 0 and self.check_winner_at(i,j):  # If the position is empty and it creates a winning move.
                    winning_moves.append((i, j))  # Add the winning move to the list.
        if winning_moves:  # If there are any winning moves found.
            return winning_moves  # Return the list of winning moves immediately.
        # Then check for opponent's threats that need blocking
        blocking_moves = self._find_opponent_threats()  # Find opponent's threats that need to be blocked.
        # Finally get neighboring moves if no urgent moves found
        neighboring_moves = self._generate_neighboring_moves()  # Generate neighboring moves around existing pieces.
        # Remove blocking moves from neighboring moves, leaving only non-blocking moves.
        other_moves = [move for move in neighboring_moves if move not in blocking_moves]
        if not blocking_moves and not other_moves and not self.moves:  # If there are no blocking or other moves and no moves have been made yet.
            center = self.size // 2  # The center of the board is a good initial move if the board is empty.
            other_moves.append((center, center))  # Add the center position as a possible move.
        return blocking_moves + other_moves  # Return the list of blocking moves followed by other possible moves.

    def _compute_heuristic_score(self):
        # First check if current player can win immediately
        for i in range(self.size):  # Iterate through each row.
            for j in range(self.size):  # Iterate through each column.
                if self.board[i][j] == 0 and self.check_winner_at(i,j):  # If the position is empty and it results in a win.
                    return 500000  # Return the highest possible score for an immediate winning move.
        score = 0  # Initialize the score to 0.
        directions = [(0, 1), (1, 0), (1, 1),
                      (1, -1)]  # Define the directions for horizontal, vertical, diagonal, and anti-diagonal.
        for i in range(self.size):  # Iterate through each row.
            for j in range(self.size):  # Iterate through each column.
                if self.board[i][j] == 0:  # Skip empty positions.
                    continue
                player = self.board[i][j]  # Get the current player's mark (either 1 or 2).
                multiplier = 1 if player == 1 else -1  # Set a multiplier based on the player's mark (1: AI, 2: opponent).
                for dr, dc in directions:  # For each direction, check the number of consecutive marks.
                    count = self._count_sequence(i, j, dr,dc)  # Count the number of consecutive marks in this direction.
                    if count < 2:  # Ignore sequences of length less than 2.
                        continue
                    # Check if the sequence is open-ended
                    is_open_start = False
                    is_open_end = False
                    start_r = i - dr  # Row before the start of the sequence.
                    start_c = j - dc  # Column before the start of the sequence.
                    end_r, end_c = i + dr * count, j + dc * count  # Row and column at the end of the sequence.
                    # Check if the start of the sequence is open (empty spot).
                    if 0 <= start_r < self.size and 0 <= start_c < self.size and self.board[start_r][start_c] == 0:
                        is_open_start = True
                    # Check if the end of the sequence is open (empty spot).
                    if 0 <= end_r < self.size and 0 <= end_c < self.size and self.board[end_r][end_c] == 0:
                        is_open_end = True
                    # Assign scores based on sequence length and whether it is open-ended or closed
                    if count == 4:  # For a sequence of four consecutive marks.
                        if is_open_start and is_open_end:
                            score += multiplier * 100000  # Open four (both ends are open).
                        elif is_open_start or is_open_end:
                            score += multiplier * 5000  # Half-open four (one end is open).
                        else:
                            score += multiplier * 1000  # Closed four (no ends are open).
                        if player == 2:  # If the opponent has a four-in-a-row.
                            score -= multiplier * 500000  # Increase penalty to block the opponent's four.
                    elif count == 3:  # For a sequence of three consecutive marks.
                        if is_open_start and is_open_end:
                            score += multiplier * 1000  # Open three (both ends are open).
                        elif is_open_start or is_open_end:
                            score += multiplier * 500  # Half-open three (one end is open).
                        else:
                            score += multiplier * 100  # Closed three (no ends are open).
                    elif count == 2:  # For a sequence of two consecutive marks.
                        if is_open_start and is_open_end:
                            score += multiplier * 100  # Open two (both ends are open).
                        elif is_open_start or is_open_end:
                            score += multiplier * 50  # Half-open two (one end is open).
                    elif count >= 5:  # For a winning move (5 or more marks in a row).
                        score += multiplier * 500000  # Assign the highest score for a winning move.
        return score  # Return the computed heuristic score.

    def evaluate(self):
        # Evaluate the board state for the Alpha-Beta algorithm
        winner = self.check_winner()  # Check if there is a winner
        if winner == 1:
            return 1000  # AI wins, assign the highest score
        if winner == 2:
            return -1000  # Opponent wins, assign the lowest score
        return self._compute_heuristic_score()  # No winner, compute the heuristic score for the current board state

    def _maximize(self, depth, alpha, beta):
        # Maximizing player (AI's move) logic for Alpha-Beta Pruning
        max_eval = float('-inf')  # Start with the lowest possible score
        best_move = None  # No best move initially
        for move in self.get_possible_moves():  # Iterate through all possible moves
            self.make_move(*move)  # Make the move
            eval_score, _ = self.alpha_beta(depth - 1, alpha, beta,
                                            False)  # Call alpha-beta pruning recursively for opponent's turn
            self.undo_move()  # Undo the move to explore other possibilities
            if eval_score > max_eval:  # If this move gives a better score
                max_eval = eval_score  # Update the best score
                best_move = move  # Update the best move
            alpha = max(alpha, eval_score)  # Update the alpha value
            if beta <= alpha:  # If the beta value is less than or equal to alpha, prune the search
                break
        return max_eval, best_move  # Return the best score and corresponding move

    def _minimize(self, depth, alpha, beta):
        # Minimizing player (human's move) logic for Alpha-Beta Pruning
        min_eval = float('inf')  # Start with the highest possible score
        best_move = None  # No best move initially
        for move in self.get_possible_moves():  # Iterate through all possible moves
            self.make_move(*move)  # Make the move
            eval_score, _ = self.alpha_beta(depth - 1, alpha, beta,
                                            True)  # Call alpha-beta pruning recursively for AI's turn
            self.undo_move()  # Undo the move to explore other possibilities
            if eval_score < min_eval:  # If this move gives a better score
                min_eval = eval_score  # Update the best score
                best_move = move  # Update the best move
            beta = min(beta, eval_score)  # Update the beta value
            if beta <= alpha:  # If the alpha value is greater than or equal to beta, prune the search
                break
        return min_eval, best_move  # Return the best score and corresponding move

    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        # Implement Alpha-Beta Pruning to find the best move
        board_hash = self._get_board_hash()  # Get a unique hash for the current board state
        tt_entry = self.transposition_table.get((board_hash, depth, maximizing_player))  # Check if the result is cached
        if tt_entry is not None:
            return tt_entry  # Return cached result if available
        if depth == 0 or self.check_winner() != 0 or self.is_board_full():
            # If maximum depth is reached, a winner is found, or the board is full, evaluate the board
            eval_score = self.evaluate()  # Evaluate the board state (calling the heuristic function)
            self.transposition_table[(board_hash, depth, maximizing_player)] = (eval_score, None)  # Cache the result
            return eval_score, None  # Return the evaluation score (no best move as it's a terminal state)
        if maximizing_player:
            result = self._maximize(depth, alpha, beta)  # Call the maximizing player's strategy (AI's turn)
        else:
            result = self._minimize(depth, alpha, beta)  # Call the minimizing player's strategy (human's turn)
        # Cache the result of the current search for future reference
        self.transposition_table[(board_hash, depth, maximizing_player)] = result
        return result

    def minimax_maximize(self, depth):
        # Maximizing player (AI's move) logic for Minimax
        max_eval = float('-inf')  # Initialize the maximum score as negative infinity
        best_move = None  # Initialize the best move as None
        for move in self.get_possible_moves():  # Iterate through all possible moves
            self.make_move(*move)  # Make the move on the board
            eval_score, _ = self.minimax(depth - 1,False)  # Recursively call minimax for the next depth with minimizing player
            self.undo_move()  # Undo the move after the evaluation
            if eval_score > max_eval:  # Update the best move if a higher evaluation score is found
                max_eval = eval_score
                best_move = move
        return max_eval, best_move  # Return the best move and its evaluation score

    def minimax_minimize(self, depth):
        # Minimizing player (human's move) logic for Minimax
        min_eval = float('inf')  # Initialize the minimum score as positive infinity
        best_move = None  # Initialize the best move as None
        for move in self.get_possible_moves():  # Iterate through all possible moves
            self.make_move(*move)  # Make the move on the board
            eval_score, _ = self.minimax(depth - 1,True)  # Recursively call minimax for the next depth with maximizing player
            self.undo_move()  # Undo the move after the evaluation
            if eval_score < min_eval:  # Update the best move if a lower evaluation score is found
                min_eval = eval_score
                best_move = move
        return min_eval, best_move  # Return the best move and its evaluation score

    def minimax(self, depth, maximizing_player):
        # Implement Alpha-Beta Pruning to find the best move
        board_hash = self._get_board_hash()  # Get a unique hash for the current board state
        tt_entry = self.transposition_table.get(
            (board_hash, depth, maximizing_player))  # Check the transposition table for a cached result
        if tt_entry is not None:
            return tt_entry  # Return cached result if available
        # Base Case: If maximum depth is reached or game over (win/loss/draw)
        if depth == 0 or self.check_winner() != 0 or self.is_board_full():
            eval_score = self.evaluate()  # Evaluate the current board state
            self.transposition_table[(board_hash, depth, maximizing_player)] = (
            eval_score, None)  # Store the result in the transposition table
            return eval_score, None
        # Recursive Case: Depending on whose turn it is, maximize or minimize the score
        if maximizing_player:
            result = self.minimax_maximize(depth)  # Maximizing player (AI's move)
        else:
            result = self.minimax_minimize(depth)  # Minimizing player (human's move)
        # Store the result in the transposition table for future lookups
        self.transposition_table[(board_hash, depth, maximizing_player)] = result
        return result


def _handle_minimax_turn(game):
    print("AI is evaluating options...")
    # First, scan for any immediate winning move
    for move in game.get_possible_moves():
        game.make_move(*move)  # Make a temporary move
        if game.check_winner_at(*move) == 1:  # Check if AI wins with this move
            print(f"AI plays winning move at {move}")
            return True  # AI wins, move is made, return True
        game.undo_move()  # Undo the move if it doesn't result in a win
    # If no immediate win, proceed with alpha-beta pruning
    _, move = game.minimax(game.max_depth, True)  # Use Minimax with Alpha-Beta pruning to find the best move
    if move:  # If a move is found
        game.make_move(*move)  # Make the best move
        print(f"AI places B at {move}")
        return True  # Move made successfully
    print("No valid moves for AI.")
    return False  # No valid move was found

def _handle_human_turn(game):
    try:
        row = int(input("Enter row: "))  # Prompt the human player to input the row
        col = int(input("Enter column: "))  # Prompt the human player to input the column
        valid_moves = game.get_possible_moves()  # Get the list of valid possible moves
        # If no moves have been made yet, the human player can only place a piece in the center
        if not game.moves:
            center = game.size // 2  # Find the center of the board (for first move)
            valid_moves = [(center, center)]  # Only allow the center as a valid move
        # Check if the entered row and column are valid moves
        if (row, col) not in valid_moves:
            print("Invalid move")  # If not a valid move, print error message and return False
            return False
        game.make_move(row, col)  # Make the move on the game board
        return True  # Return True to indicate that the move was successful
    except ValueError:  # Handle the case where input cannot be converted to an integer
        print("Please enter valid numbers.")  # Inform the player to enter valid numbers
        return False  # Return False to indicate the move was not successful


def _check_game_end(game):
    winner = game.check_winner()  # Check if there’s a winner (returns 1 for player B, 2 for player W, or 0 for no winner)
    if winner:  # If a winner is found
        game.print_board()  # Print the board to show the final state
        print(
            f"Player {'B' if winner == 1 else 'W'} wins!")  # Announce the winner based on the player's color (1 for B, 2 for W)
        return True  # The game ends
    if game.is_board_full():  # If the board is full, it's a draw
        game.print_board()  # Print the board to show the final state
        print("It's a draw!")  # Announce the draw
        return True  # The game ends
    return False  # The game is still ongoing

def display_menu():
    """Display game menu and get user choice."""
    print("\n===== Gomoku Menu =====")
    print("1. Human vs AI (Minimax)")
    print("2. Human vs AI (Alpha-Beta)")
    print("3. AI vs AI (Minimax vs Alpha-Beta)")
    print("4. Exit")
    return input("Choose an option (1-4): ").strip()  # Get user input and return the option they choose


def human_vs_ai(game, use_alpha_beta=True):
    """Human vs AI game loop."""
    # Print a message showing whether the AI is using Alpha-Beta Pruning or Minimax
    print(f"\nHuman (W) vs AI (B) using {'Alpha-Beta' if use_alpha_beta else 'Minimax'}")
    # Set AI to play first (AI is player 1, human is player 2)
    game.current_player = 1  # AI starts first
    # Start the game loop that continues until the game ends (win or draw)
    while True:
        # Print the current state of the board
        game.print_board()
        if game.current_player == 1:  # AI's turn
            print("AI is thinking...")
            # First, check if there is an immediate winning move for AI
            if use_alpha_beta:
                # If using Alpha-Beta Pruning, first check for immediate winning moves
                for move in game.get_possible_moves():
                    game.make_move(*move)  # Try the move
                    if game.check_winner_at(*move) == 1:  # Check if this move wins the game
                        print(f"AI plays winning move at {move}")  # Announce the move
                        break  # Exit the loop once the winning move is found
                    game.undo_move()  # Undo the move if it doesn't win
                else:
                    # If no immediate win, perform Alpha-Beta pruning
                    _, move = game.alpha_beta(game.max_depth, float('-inf'), float('inf'), True)
                    if move:  # If a valid move is found
                        game.make_move(*move)  # Make the move
                        print(f"AI places B at {move}")  # Announce the move
                    else:
                        print("No valid moves for AI.")  # If no valid moves, end the game
                        break  # Exit the loop
            else:
                # If not using Alpha-Beta pruning, use Minimax to evaluate the best move
                for move in game.get_possible_moves():
                    game.make_move(*move)  # Try the move
                    if game.check_winner_at(*move) == 1:  # Check if this move wins the game
                        print(f"AI plays winning move at {move}")  # Announce the winning move
                        break  # Exit the loop once the winning move is found
                    game.undo_move()  # Undo the move if it doesn't win
                else:
                    # If no immediate win, perform Minimax to find the best move
                    _, move = game.minimax(game.max_depth, True)
                    if move:  # If a valid move is found
                        game.make_move(*move)  # Make the move
                        print(f"AI places B at {move}")  # Announce the move
                    else:
                        print("No valid moves for AI.")  # If no valid moves, end the game
                        break  # Exit the loop
        else:  # Human's turn
            # Ask the human player to input a valid row and column
            try:
                row = int(input("Enter row: "))
                col = int(input("Enter column: "))
                # If the human moves, make sure the move is valid
                if not game.make_move(row, col):
                    print("Invalid move! Try again.")  # If the move is invalid, ask the human to try again
                    continue
            except ValueError:
                print("Please enter valid numbers.")  # Handle non-numeric input gracefully
                continue  # Prompt the human to input again
        # After each turn, check if the game is over (either win or draw)
        winner = game.check_winner()  # Check if there is a winner
        if winner:  # If there is a winner
            game.print_board()  # Print the final state of the board
            print(f"Player {'B' if winner == 1 else 'W'} wins!")  # Print the winning player
            break  # End the game
        if game.is_board_full():  # If the board is full (no empty spaces left)
            game.print_board()  # Print the final state of the board
            print("It's a draw!")  # Print that the game is a draw
            break  # End the game


def ai_vs_ai(game):
    """AI vs AI game loop (Minimax vs Alpha-Beta)."""
    # Print the message showing that two AIs are playing against each other, one using Minimax and the other using Alpha-Beta.
    print("\nAI (Minimax as B) vs AI (Alpha-Beta as W)")
    # Set the current player to AI using Minimax (player 1 starts first).
    game.current_player = 1  # Minimax starts first
    # Start the game loop that continues until the game ends (win or draw).
    while True:
        # Print the current state of the board.
        game.print_board()
        # Set the name of the current player based on their ID: Minimax (B) or Alpha-Beta (W).
        current_player_name = 'Minimax (B)' if game.current_player == 1 else 'Alpha-Beta (W)'
        print(f"{current_player_name} is thinking...")  # Announce the current AI's thinking.
        # Initialize a flag to track if the current player makes a winning move.
        made_move = False
        # Check if the current player can win immediately with any possible move.
        for move in game.get_possible_moves():
            game.make_move(*move)  # Try the current move.
            # Check if the move results in a win for the current player.
            if game.check_winner_at(*move) == game.current_player:
                print(f"{current_player_name} plays winning move at {move}")  # Announce the winning move.
                made_move = True  # Mark that the move is made.
                break  # Exit the loop since the winning move has been found.
            game.undo_move()  # Undo the move if it doesn't win.
        if not made_move:  # If no winning move was found for the current player.
            # Check if the opponent can win on the next move and block that move.
            opponent = 3 - game.current_player  # Get the opponent's player ID (1 becomes 2 and vice versa).
            blocking_move = None  # Initialize a variable to store the blocking move, if any.
            for move in game.get_possible_moves():
                game.make_move(*move)  # Try the current move.
                # If the move results in a win for the opponent, block it.
                if game.check_winner_at(*move) == opponent:
                    blocking_move = move  # Store the blocking move.
                    game.undo_move()  # Undo the move after checking.
                    break  # Exit the loop once a blocking move is found.
                game.undo_move()  # Undo the move if it doesn't block the opponent's win.
            if blocking_move:  # If a blocking move was found.
                game.make_move(*blocking_move)  # Make the blocking move.
                print(f"{current_player_name} blocks opponent's winning move at {blocking_move}")  # Announce the block.
            else:
                # No immediate threats from the opponent, proceed with normal AI move.
                if game.current_player == 1:
                    # If it's Minimax's turn, run the Minimax algorithm to get the best move.
                    _, move = game.minimax(game.max_depth, True)
                else:
                    # If it's Alpha-Beta's turn, run the Alpha-Beta pruning algorithm.
                    _, move = game.alpha_beta(game.max_depth, float('-inf'), float('inf'), False)

                if move:  # If a valid move is found by the AI.
                    game.make_move(*move)  # Make the chosen move.
                    print(
                        f"{current_player_name} places {'B' if game.current_player == 1 else 'W'} at {move}")  # Announce the move.
                else:
                    print("No valid moves for AI.")  # If no valid moves, end the game.
                    break  # Exit the loop and end the game.
        # After the move, check if there is a winner.
        winner = game.check_winner()
        if winner:  # If there's a winner.
            game.print_board()  # Print the final board.
            print(f"Player {'B' if winner == 1 else 'W'} wins!")  # Announce the winner (B or W).
            break  # End the game.
        # If the board is full, it's a draw.
        if game.is_board_full():
            game.print_board()  # Print the final board.
            print("It's a draw!")  # Announce the draw.
            break  # End the game.

def main():
    """Main program entry point."""
    # Prompt the user to enter a board size. Default is 15 if no input is given.
    size = int(input("Enter board size (default 15): ") or 15)
    # Start an infinite loop to display the main menu and process user input.
    while True:
        # Call the function to display the menu and capture the user's choice.
        choice = display_menu()
        # Check the user's choice and perform the appropriate action.
        if choice == "1":
            # Option 1: Start a game with human vs AI (without Alpha-Beta).
            game = Gomoku(size=size)  # Create a new Gomoku game instance with the specified board size.
            human_vs_ai(game, use_alpha_beta=False)  # Call the function to start a human vs AI game without Alpha-Beta pruning.
        elif choice == "2":
            # Option 2: Start a game with human vs AI (with Alpha-Beta).
            game = Gomoku(size=size)  # Create a new Gomoku game instance.
            human_vs_ai(game, use_alpha_beta=True)  # Call the function to start a human vs AI game with Alpha-Beta pruning.
        elif choice == "3":
            # Option 3: Start a game with AI vs AI (Minimax vs Alpha-Beta).
            game = Gomoku(size=size)  # Create a new Gomoku game instance.
            ai_vs_ai(game)  # Call the function to start an AI vs AI game.
        elif choice == "4":
            # Option 4: Exit the program.
            print("Exiting...")  # Display a message to indicate the program is exiting.
            break  # Exit the while loop, ending the program.
        else:
            # If the user enters an invalid choice, print an error message.
            print("Invalid choice! Try again.")

# This conditional ensures that the main function runs only if the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    main()
