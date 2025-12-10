import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from preprocessing import Preprocessing

class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG (Normal & PVC)")
        self.root.geometry("1280x720")
        
        # --- UPDATED INSTANTIATION ---
        self.logic = Preprocessing()
        
        # Variables to store data for plotting
        self.raw_signal = None
        self.processed_signal = None

        self._setup_ui()

    def _setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        # --- Controls Area ---
        controls_frame = ttk.LabelFrame(self.main_frame, text="Step 1 Controls", padding=10)
        controls_frame.pack(fill="x", pady=5)

        self.btn_load = ttk.Button(controls_frame, text="1. Load Signal", command=self.handle_load)
        self.btn_load.pack(side="left", padx=5)

        self.btn_process = ttk.Button(controls_frame, text="2. Apply Filter & Normalize", command=self.handle_process, state="disabled")
        self.btn_process.pack(side="left", padx=5)

        # --- Plotting Area ---
        plot_frame = ttk.Frame(self.main_frame)
        plot_frame.pack(fill="both", expand=True, pady=10)

        # Setup Matplotlib Figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def handle_load(self):
        """User clicks Load Button"""
        file_path = filedialog.askopenfilename(filetypes=[("Text/CSV", "*.txt *.csv"), ("All Files", "*.*")])
        if not file_path:
            return

        try:
            # Call logic from preprocessing.py
            self.raw_signal = self.logic.load_data(file_path)
            
            # Update GUI
            self.ax1.clear()
            self.ax1.plot(self.raw_signal, color='blue', linewidth=0.8)
            self.ax1.set_title("Raw Input Signal")
            self.ax1.grid(True)
            self.canvas.draw()
            
            self.btn_process.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def handle_process(self):
        """User clicks Process Button"""
        if self.raw_signal is None:
            return

        try:
            # Call logic from preprocessing.py
            self.processed_signal = self.logic.apply_processing(self.raw_signal)

            # Update GUI
            self.ax2.clear()
            self.ax2.plot(self.processed_signal, color='green', linewidth=0.8)
            self.ax2.set_title("Step 1 Result: Filtered (0.5-40Hz) & Normalized")
            self.ax2.grid(True)
            self.canvas.draw()
            
            messagebox.showinfo("Success", "Step 1 Complete!")
            
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))