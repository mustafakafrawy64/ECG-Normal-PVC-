import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from gui import SignalApp


if __name__ == "__main__":
    root = tk.Tk()

    app = SignalApp(root)


    root.mainloop()
