import numpy as np
from scipy.signal import butter, filtfilt

class Preprocessing:
    
    def __init__(self):
        self.sampling_rate = 360  # Hz (MIT_BIH standard)

    def load_data(self, file_path): #load dataset and returns numpy array of the signal.
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            content = content.replace('|', ' ').replace(',', ' ')
            
            # Convert to numpy array
            data = np.fromstring(content, sep=' ')
            
            # Safety check: Ensure we have data
            if data.size == 0:
                raise ValueError("File appears to be empty or contains no valid numbers.")
                
            return data

        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

    def apply_processing(self, raw_signal): # #Applies the Band Pass Filter and Normalization.
     
        # 1. Band Pass Filter (Butterworth)
        b, a = butter(N=2, Wn=[0.5, 40], btype='bandpass', fs=self.sampling_rate)
        
        # 'filtfilt' is the built-in zero-phase filter (prevents peak shifting)
        filtered_signal = filtfilt(b, a, raw_signal)

        # 2. Normalization (Min-Max)
        min_val = np.min(filtered_signal)
        max_val = np.max(filtered_signal)
        
        # Avoid division by zero if signal is flat
        if max_val - min_val == 0:
            return filtered_signal
            
        normalized_signal = (filtered_signal - min_val) / (max_val - min_val)
        
        return normalized_signal