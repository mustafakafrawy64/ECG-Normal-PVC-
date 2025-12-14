import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from preprocessing import Preprocessing
from feature_extraction import FeatureExtraction
from classifier import KNNClassifier
import numpy as np
import os


class SignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG (Normal & PVC)")
        self.root.geometry("1280x720")

        # Core logic modules
        self.logic = Preprocessing()
        self.fe = FeatureExtraction(ac_keep=50, dct_keep=20)

        # Classifier instance
        self.classifier = KNNClassifier(k=3)

        # Data holders
        self.raw_signal = None
        self.processed_signal = None
        self.features = None

        self._setup_ui()

    def _setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill="both", expand=True)

        controls_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        controls_frame.pack(fill="x", pady=5)

        self.btn_load = ttk.Button(
            controls_frame, text="1. Load Signal", command=self.handle_load
        )
        self.btn_load.pack(side="left", padx=5)

        self.btn_process = ttk.Button(
            controls_frame,
            text="2. Filter & Normalize",
            command=self.handle_process,
            state="disabled",
        )
        self.btn_process.pack(side="left", padx=5)

        self.btn_features = ttk.Button(
            controls_frame,
            text="3. Extract Features (AC + DCT)",
            command=self.handle_features,
            state="disabled",
        )
        self.btn_features.pack(side="left", padx=5)

        # Train & Classify button (disabled until features extracted)
        self.btn_classify = ttk.Button(
            controls_frame,
            text="4. Train & Classify (KNN)",
            command=self.handle_classify,
            state="disabled",
        )
        self.btn_classify.pack(side="left", padx=5)

        # Build training CSV button (always available)
        self.btn_build_csv = ttk.Button(
            controls_frame,
            text="5. Build Training CSV",
            command=self.handle_build_training_csv,
        )
        self.btn_build_csv.pack(side="left", padx=5)

        plot_frame = ttk.Frame(self.main_frame)
        plot_frame.pack(fill="both", expand=True, pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -------------------------
    # Step 1 - Load raw signal
    # -------------------------
    def handle_load(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.raw_signal = self.logic.load_data(file_path)

            self.ax1.clear()
            self.ax1.plot(self.raw_signal, color="blue", linewidth=0.8)
            self.ax1.set_title("Raw Input Signal")
            self.ax1.grid(True)
            self.canvas.draw()

            # enable processing after load
            self.btn_process.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -------------------------
    # Step 1.1 - Process
    # -------------------------
    def handle_process(self):
        if self.raw_signal is None:
            return

        try:
            self.processed_signal = self.logic.apply_processing(self.raw_signal)

            self.ax2.clear()
            self.ax2.plot(self.processed_signal, color="green", linewidth=0.8)
            self.ax2.set_title("Step 1 Result: Filtered & Normalized")
            self.ax2.grid(True)
            self.canvas.draw()

            # enable feature extraction
            self.btn_features.config(state="normal")

            messagebox.showinfo("Success", "Step 1 Complete!")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    # -------------------------
    # Step 2 - Feature Extraction (AC + DCT)
    # -------------------------
    def handle_features(self):
        if self.processed_signal is None:
            return

        try:
            self.features = self.fe.extract_features(self.processed_signal)

            ac = self.features.get("autocorrelation", [])
            dct_feats = self.features.get("dct_features", [])

            # enable classify button now that features exist
            self.btn_classify.config(state="normal")

            messagebox.showinfo(
                "Step 2 Complete",
                f"Autocorrelation Coeffs: {len(ac)}\n"
                f"DCT Features: {len(dct_feats)}\n\n"
                f"Feature extraction successful!",
            )

        except Exception as e:
            messagebox.showerror("Feature Extraction Error", str(e))

    # ----------------------------------------------------------
    # Step 3 — Train & Classify using the KNN classifier
    # ----------------------------------------------------------
    def handle_classify(self):
        if self.features is None:
            messagebox.showwarning("No features", "Please run feature extraction first.")
            return

        # choose training file (CSV or .npz)
        file_path = filedialog.askopenfilename(
            title="Select training data CSV or .npz",
            filetypes=[("CSV", "*.csv"), ("NumPy npz", "*.npz"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            # Load training data
            if file_path.lower().endswith(".npz"):
                archive = np.load(file_path, allow_pickle=True)
                X = archive["X"]
                y = archive["y"]
            else:
                X, y = KNNClassifier.load_training_csv(file_path)

            # Train
            self.classifier.fit(X, y)

            # Use DCT features for classification (ensure 1D)
            dct_feats = np.array(self.features["dct_features"], dtype=float).ravel()

            pred_label, votes, neigh_labels, neigh_dists = self.classifier.predict(
                dct_feats
            )

            # Confidence percentage (KNN vote ratio)
            total_votes = sum(votes.values())
            confidence = (votes.get(pred_label, 0) / total_votes) * 100 if total_votes > 0 else 0.0

            messagebox.showinfo(
                "Classification Result",
                f"Predicted Class: {pred_label}\n"
                f"Accuracy: {confidence:.2f}%\n\n"
                f"Votes: {votes}\n\n"
                f"Neighbors: {neigh_labels}",
            )


        except Exception as e:
            messagebox.showerror("Classification Error", str(e))

    # ----------------------------------------------------------
    # Step 5 — Build Training CSV from raw dataset files (DCT only)
    # ----------------------------------------------------------
    def handle_build_training_csv(self):
        # Allow multiple files to be selected
        file_paths = filedialog.askopenfilenames(
            title="Select raw ECG dataset files",
            filetypes=[("Text files", "*.txt *.csv"), ("All files", "*.*")],
        )
        if not file_paths:
            return

        # Ask output CSV path
        out_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")],
            title="Save training CSV as...",
            initialfile="training_data.csv",
        )
        if not out_path:
            return

        try:
            # Open output file (write mode). We will write rows as: dct1,dct2,...,dctN,label
            with open(out_path, "w") as out:
                # Optional header (commented out). Uncomment if you want a header row.
                # header = ",".join([f"dct{i+1}" for i in range(self.fe.dct_keep)]) + ",label\n"
                # out.write(header)

                for path in file_paths:
                    # prompt for label for this file
                    # default suggestion: filename-derived label (e.g., 'Normal_Train' -> 'normal')
                    filename = os.path.basename(path)
                    suggested = self._suggest_label_from_filename(filename)
                    label = simpledialog.askstring(
                        "Label",
                        f"Enter label for:\n{path}\n\nSuggested: {suggested}\n\nExamples: normal, pvc",
                        initialvalue=suggested,
                    )
                    if not label:
                        # skip file if user cancels label
                        messagebox.showwarning("Skipped", f"Skipped file: {path}")
                        continue

                    # Load raw signal
                    raw = self.logic.load_data(path)

                    # Preprocess and extract features
                    processed = self.logic.apply_processing(raw)
                    feats = self.fe.extract_features(processed)
                    dct_feats = np.array(feats.get("dct_features", []), dtype=float).ravel()

                    # Ensure consistent length by padding/truncating to fe.dct_keep
                    if dct_feats.size < self.fe.dct_keep:
                        padded = np.zeros(self.fe.dct_keep, dtype=float)
                        padded[: dct_feats.size] = dct_feats
                        dct_feats = padded
                    elif dct_feats.size > self.fe.dct_keep:
                        dct_feats = dct_feats[: self.fe.dct_keep]

                    # Write row
                    row = ",".join(map(str, dct_feats.tolist())) + f",{label}\n"
                    out.write(row)

            messagebox.showinfo(
                "Training CSV Ready", f"Training CSV successfully created:\n{out_path}"
            )

        except Exception as e:
            messagebox.showerror("Error Building CSV", str(e))

    def _suggest_label_from_filename(self, filename: str) -> str:
        name = filename.lower()
        if "normal" in name:
            return "normal"
        if "pvc" in name or "pvcs" in name:
            return "pvc"
        # fallback - just return filename without extension
        return os.path.splitext(filename)[0]

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = SignalApp(root)
#     root.mainloop()
