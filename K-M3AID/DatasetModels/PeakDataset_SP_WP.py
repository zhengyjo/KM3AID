import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PeakDataset(Dataset):
    def __init__(self, files, CLMAConfig):
        self.files = files
        self.cnmr_path = CLMAConfig.cnmr_path
        self.device = CLMAConfig.device
        self.temperature = CLMAConfig.mr1_element_temperature
        self.diff_temperature = CLMAConfig.mr1_element_diff_temperature 

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        return self.files[idx]

    def __getitem__(self, index):
        file_path = self.cnmr_path + self.files[index]

        # Load NMR data from file with optimized dtype specification
        df = pd.read_csv(file_path, dtype={'atom': int, 'ppm': float})
        df.sort_values(by='atom', inplace=True)

        # Extract NMR data and apply the preprocessing transform
        peak = df['ppm'].values.tolist()
        pindex = df['pattern_index'].values.tolist() 
        peak_tensor = torch.tensor(peak, dtype=torch.float32)
        pindex_tensor = torch.tensor(pindex, dtype=torch.float32) 
        return peak_tensor, pindex_tensor

    def collate_fn(self, batch):
        # Separate preprocessed NMR data and raw NMR data
        ppm, pattern  = zip(*batch)
        # Calculate the number of peaks in each sample
        num_peaks = [len(p) for p in pattern]
        num_peaks = torch.tensor(num_peaks)
        # Stack the preprocessed NMR data along a new dimension (batch dimension)
        pattern = torch.cat(pattern, dim=0)
        ppm = torch.cat(ppm, dim=0)
        ppm = ppm.view(-1, 1)

        ppm_diff = torch.abs(ppm - ppm.T)
 
        #ppm_diff += self.temperature 
        #ppm_diff = 1/ppm_diff * self.diff_temperature  
        ppm_diff = (ppm_diff <=  self.diff_temperature).float()  
        
        return  { 'peak': ppm, 'pattern': pattern, 'diff': ppm_diff, 'num_peaks': num_peaks}





