import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from preprocessing import Preprocessing
from feature_extraction import FeatureExtraction   # <-- ADDED


class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG (Normal & PVC)")
        self.root.geometry("1280x720")

        self.logic = Preprocessing()
        self.fe = FeatureExtraction(ac_keep=50, dct_keep=20)   # <-- ADDED

        self.raw_signal = None
        self.processed_signal = None

        self._setup_ui()

    def _setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        controls_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        controls_frame.pack(fill="x", pady=5)

        self.btn_load = ttk.Button(controls_frame, text="1. Load Signal", command=self.handle_load)
        self.btn_load.pack(side="left", padx=5)

        self.btn_process = ttk.Button(controls_frame, text="2. Filter & Normalize",
                                      command=self.handle_process, state="disabled")
        self.btn_process.pack(side="left", padx=5)

        self.btn_features = ttk.Button(controls_frame, text="3. Extract Features (AC + DCT)",
                                       command=self.handle_features, state="disabled")   # <-- ADDED
        self.btn_features.pack(side="left", padx=5)

        plot_frame = ttk.Frame(self.main_frame)
        plot_frame.pack(fill="both", expand=True, pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def handle_load(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.raw_signal = self.logic.load_data(file_path)

            self.ax1.clear()
            self.ax1.plot(self.raw_signal, color='blue', linewidth=0.8)
            self.ax1.set_title("Raw Input Signal")
            self.ax1.grid(True)
            self.canvas.draw()

            self.btn_process.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def handle_process(self):
        if self.raw_signal is None:
            return

        try:
            self.processed_signal = self.logic.apply_processing(self.raw_signal)

            self.ax2.clear()
            self.ax2.plot(self.processed_signal, color='green', linewidth=0.8)
            self.ax2.set_title("Step 1 Result: Filtered & Normalized")
            self.ax2.grid(True)
            self.canvas.draw()

            self.btn_features.config(state="normal")   # <-- ENABLE STEP 2

            messagebox.showinfo("Success", "Step 1 Complete!")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    # ----------------------------------------------------------
    # STEP 2 HANDLER (AC + DCT)
    # ----------------------------------------------------------
    def handle_features(self):
        if self.processed_signal is None:
            return

        try:
            features = self.fe.extract_features(self.processed_signal)

            ac = features["autocorrelation"]
            dct_feats = features["dct_features"]

            messagebox.showinfo(
                "Step 2 Complete",
                f"Autocorrelation Coeffs: {len(ac)}\n"
                f"DCT Features: {len(dct_feats)}\n\n"
                f"Feature extraction successful!"
            )

        except Exception as e:
            messagebox.showerror("Feature Extraction Error", str(e))
