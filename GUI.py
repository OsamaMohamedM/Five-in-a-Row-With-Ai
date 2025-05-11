import tkinter as tk
from tkinter import messagebox
from Gomoku import *

class GomokuApp:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()
        self.game = None
        self.canvas = None
        self.buttons = []
        self.status_label = None
        self.gameMode = ""
        self.chooseGameMode()

    def chooseGameMode(self):
        self.mode_window = tk.Toplevel(self.root)
        self.mode_window.title("Choose Game Mode")
        self.mode_window.geometry("400x400")

        label = tk.Label(self.mode_window, text="Choose Game Mode", font=("Helvetica", 18))
        label.pack(pady=10)

        btnHumanVsAi = tk.Button(self.mode_window, text="Human vs AI", font=("Helvetica", 14),
                                 command=lambda: self.selectMode("human_vs_ai"))
        btnHumanVsAi.pack(pady=10)

        btnAiVsAi = tk.Button(self.mode_window, text="AI vs AI", font=("Helvetica", 14),
                              command=lambda: self.selectMode("ai_vs_ai"))
        btnAiVsAi.pack(pady=10)

    def selectMode(self, mode):
        self.gameMode = mode
        self.mode_window.destroy()
        self.root.deiconify()
        self.openMainGameWindow()

    def openMainGameWindow(self):
        self.game = Gomoku(size=15)
        self.buttons = []

        self.status_label = tk.Label(self.root, text="Starting game...", font=("Helvetica", 14))
        self.status_label.pack()

        self.canvas = tk.Canvas(self.root, width=self.game.size * 30, height=self.game.size * 30, bg="white")
        self.canvas.pack()

        self.draw_grid()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        back_button = tk.Button(self.root, text="Back to Game Modes", font=("Helvetica", 14), command=self.back_to_game_modes)
        back_button.pack(pady=10)

        if self.gameMode == "human_vs_ai" and self.game.current_player == 1:
            self.status_label.config(text="AI (Minimax) thinking...")
            self.root.after(500, lambda: self.ai_turn(use_alpha_beta=False))
        elif self.gameMode == "ai_vs_ai" and self.game.current_player == 1:
            self.root.after(500, self.ai_vs_ai_loop)

    def draw_grid(self):
        for i in range(self.game.size):
            for j in range(self.game.size):
                x0 = j * 30
                y0 = i * 30
                x1 = x0 + 30
                y1 = y0 + 30
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline="gray", fill="white")
                self.buttons.append((rect, i, j))

    def on_canvas_click(self, event):
        if self.game.current_player != 2:
            return

        row = event.y // 30
        col = event.x // 30

        if not self.is_adjacent_to_move(row, col):
            messagebox.showinfo("Invalid Move", "You must play adjacent to an existing move.")
            return

        if self.game.make_move(row, col):
            self.update_board()
            if self.check_game_end():
                return
            self.status_label.config(text="AI thinking...")
            self.root.after(500, self.ai_turn, False)

    def is_adjacent_to_move(self, row, col):
        if self.game.board[row][col] != 0:
            return False

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.game.size and 0 <= nc < self.game.size:
                if self.game.board[nr][nc] != 0:
                    return True
        return False

    def update_board(self):
        for rect, i, j in self.buttons:
            value = self.game.board[i][j]
            color = "black" if value == 1 else "purple" if value == 2 else "white"
            self.canvas.itemconfig(rect, fill=color)

    def ai_turn(self, use_alpha_beta=False):
        algo_name = "Alpha-Beta" if use_alpha_beta else "Minimax"
        print(f"{algo_name} is thinking...")

        player = self.game.current_player

        for move in self.game.get_possible_moves():
            self.game.make_move(*move)
            if self.game.check_winner_at(*move) == player:
                self.update_board()
                self.status_label.config(text=f"{algo_name} wins!")
                messagebox.showinfo("Game Over", f"{algo_name} ({'B' if player == 1 else 'W'}) wins!")
                return True
            self.game.undo_move()

        opponent = 3 - player
        for move in self.game.get_possible_moves():
            self.game.make_move(*move)
            if self.game.check_winner_at(*move) == opponent:
                self.game.undo_move()
                self.game.make_move(*move)
                self.update_board()
                print(f"{algo_name} blocks opponent at {move}")
                return self.check_game_end()
            self.game.undo_move()

        if use_alpha_beta:
            _, move = self.game.alpha_beta(self.game.max_depth, float('-inf'), float('inf'), player == 1)
        else:
            _, move = self.game.minimax(self.game.max_depth, player == 1)

        if move:
            self.game.make_move(*move)
            self.update_board()
            if self.check_game_end():
                return True
        else:
            messagebox.showinfo("Game Over", "No valid moves.")
            return True

        return False

    def ai_vs_ai_loop(self):
        if self.game.current_player == 1:
            self.status_label.config(text="Minimax turn...")
            print("Minimax is playing...")
            game_over = self.ai_turn(use_alpha_beta=False)
        else:
            self.status_label.config(text="Alpha-Beta turn...")
            print("Alpha-Beta is playing...")
            game_over = self.ai_turn(use_alpha_beta=True)

        if not game_over:
            self.root.after(500, self.ai_vs_ai_loop)

    def check_game_end(self):
        winner = self.game.check_winner()
        if winner == 1:
            msg = "Minimax (Black) wins!" if self.gameMode == "ai_vs_ai" else "AI (Black) wins!"
            self.status_label.config(text=msg)
            messagebox.showinfo("Game Over", msg)
            return True
        elif winner == 2:
            msg = "Alpha-Beta (purple) wins!" if self.gameMode == "ai_vs_ai" else "You (purple) win!"
            self.status_label.config(text=msg)
            messagebox.showinfo("Game Over", msg)
            return True
        elif self.game.is_board_full():
            self.status_label.config(text="Draw!")
            messagebox.showinfo("Game Over", "It's a draw!")
            return True
        return False

    def back_to_game_modes(self):
        self.root.withdraw()
        for widget in self.root.winfo_children():
            widget.destroy()
        self.chooseGameMode()

if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuApp(root)
    root.mainloop()
