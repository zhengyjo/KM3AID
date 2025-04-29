import torch

class PeakEncoder(torch.nn.Module):
    def __init__(self, emb_dim=128, cnmr_type=4):
        super(PeakEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.cnmr_type = cnmr_type

        # Linear layer for the value encoding
        self.cnmr_peak_embedding = torch.nn.Linear(1, emb_dim)

        # Embedding layer for the cnmr_type encoding
        self.cnmr_type_embedding = torch.nn.Embedding(cnmr_type, emb_dim)

        # Readout layers
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, batch):
        cnmr_peak = batch['peak']
        cnmr_type = batch['pattern'] - 1

        # Combine the linear encoding with the edge_embedding1 encoding using concatenation
        cnmr_peak_encoding = self.cnmr_peak_embedding(cnmr_peak)
        cnmr_type_encoding = self.cnmr_type_embedding(cnmr_type.long())
        final_encoding = torch.cat([cnmr_peak_encoding, cnmr_type_encoding], dim=-1)
        
        # Apply readout layers
        output_encoding = self.readout(final_encoding)

        return output_encoding

