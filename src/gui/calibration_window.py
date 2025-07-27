import tkinter as tk

class CalibrationWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Calibration")
        self.geometry("400x400")
        self.grid_points = [(x, y) for x in range(3) for y in range(3)]
        self.progress = 0
        self.label = tk.Label(self, text="Calibration in progress...")
        self.label.pack()

    def show_next_point(self):
        if self.progress < len(self.grid_points):
            self.label.config(text=f"Look at point: {self.grid_points[self.progress]}")
            self.progress += 1
        else:
            self.label.config(text="Calibration complete!")
