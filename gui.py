import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox


class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG (Normal & PVC)")
        self.root.geometry("1280x720")

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill="both", expand=True)


