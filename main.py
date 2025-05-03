import copy

BOARD_SIZE = 15
DEPTH_SIZE = 3

# Directions: Right, Down, Left, Up, Diagonals (4 directions)
dx = [1, 0, -1, 0, -1, -1, 1, 1]
dy = [0, -1, 0, 1, -1, 1, -1, 1]


def is_valid(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def is_winner(board, player):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] != player:
                continue
            for i in range(8):
                count = 1
                for j in range(1, 5):
                    nx, ny = x + dx[i] * j, y + dy[i] * j
                    if is_valid(nx, ny) and board[nx][ny] == player:
                        count += 1
                    else:
                        break
                for j in range(1, 5):
                    nx, ny = x - dx[i] * j, y - dy[i] * j
                    if is_valid(nx, ny) and board[nx][ny] == player:
                        count += 1
                    else:
                        break
                if count >= 5:
                    return True
    return False


def is_draw(board):
    for row in board:
        if '.' in row:
            return False
    winner = is_winner(board, 'W') or is_winner(board, 'B')
    return not winner


def initialize_board():
    return [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def print_board(board):
    for row in board:
        print(" ".join(row))

def get_candidate_moves(board):
    candidates = set()
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] != '.':
                for d in range(8):
                    nx, ny = x + dx[d], y + dy[d]
                    if is_valid(nx, ny) and board[nx][ny] == '.':
                        candidates.add((nx, ny))
    # First move
    if not candidates:
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                candidates.add((x, y))
    return list(candidates)

def get_human_move(board):
    vaild_node = get_candidate_moves(board)
    while True:
        x, y = map(int, input("Enter your move (row and column): ").split())
        if (x, y) in vaild_node:
            return x, y
        else:
            print("Invalid input. Please enter numbers between 0 and {}.".format(BOARD_SIZE - 1))


def make_move(board, x, y, player):
    if board[x][y] == '.':
        board[x][y] = player
        return True
    return False


def evaluate_board(board, player):
    opponent = 'W' if player == 'B' else 'B'
    score = 0

    def count_consecutive(x, y, dx, dy, player):
        count = 0
        for i in range(5):
            nx, ny = x + dx * i, y + dy * i
            if is_valid(nx, ny) and board[nx][ny] == player:
                count += 1
            else:
                break
        return count

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x][y] == '.':
                for d in range(8):
                    score += count_consecutive(x, y, dx[d], dy[d], player)
                    score -= count_consecutive(x, y, dx[d], dy[d], opponent)
    return score



def minimax(board, depth, is_maximizing, player):

    opponent = 'W' if player == 'B' else 'B'

    if depth == 0 or is_winner(board, player) or is_winner(board, opponent) or is_draw(board):
        return evaluate_board(board, player), None

    best_move = None
    moves = get_candidate_moves(board)
    if not moves:
        moves = [(BOARD_SIZE // 2, BOARD_SIZE // 2)]  # fallback center

    if is_maximizing:
        max_eval = float('-inf')
        for x, y in moves:
            new_board = copy.deepcopy(board)
            new_board[x][y] = player
            eval_score, _ = minimax(new_board, depth - 1, False, player)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = (x, y)
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
        return min_eval, best_move



def minimax_move(board, depth, player='B'):
    _, move = minimax(board, depth, True, player)
    return move



def display_menu():
    print("==== Gomoku Game ====")
    print("1. Human vs AI (Minimax)")
    print("2. AI vs AI (Minimax vs Alpha-Beta)")
    print("3. Exit")
    choice = input("Choose an option (1-3): ")
    return choice


def main():
    choice = display_menu()
    board = initialize_board()
    print_board(board)
    player_turn = 'W'  # Human
    ai_turn = 'B'  # AI
    winner = False
    while not is_draw(board):
        if player_turn == 'W':
            x, y = get_human_move(board)
            make_move(board, x, y, player_turn)
            winner = is_winner(board, player_turn)
        else:
            print("AI is thinking...")
            x, y = minimax_move(board , DEPTH_SIZE , 'B')  # Placeholder
            make_move(board, x, y, ai_turn)
            print(f"AI chose: ({x}, {y})")
            winner = is_winner(board, player_turn)
        print_board(board)

        if winner:
            print(f"Player {player_turn if player_turn == 'W' else ai_turn} wins!")
            return
        if player_turn == 'W':
            player_turn = 'B'
        else:
            player_turn = 'W'
    print("Game is a draw!")

main()
