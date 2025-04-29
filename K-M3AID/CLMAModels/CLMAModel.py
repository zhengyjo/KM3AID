from torch import nn
import torch.nn.functional as F
from CLMAModels.Encoder import Encoder
from CLMAModels.Projection import Projection
import torch

class CLMAModel(nn.Module):
    def __init__(
        self,
        mr1_model,
        mr1_element_model,
        mr2_model,
        CLMAConfig
    ):
        super().__init__()
        self.mr1_encoder = Encoder(mr1_model, CLMAConfig.mr1_model_trainable)
        self.mr1_element_encoder = Encoder(mr1_element_model, CLMAConfig.mr1_element_model_trainable)
        self.mr2_encoder = Encoder(mr2_model, CLMAConfig.mr2_model_trainable)
 
        self.mr1_name = CLMAConfig.mr1_name
        self.mr1_element_name = CLMAConfig.mr1_element_name
        self.mr2_name = CLMAConfig.mr2_name

        self.temperature = CLMAConfig.temperature
        self.device = CLMAConfig.device
        self.batch_size = CLMAConfig.batch_size

        self.mr1_projection = Projection(CLMAConfig.mr1_model_embedding, CLMAConfig.projection_dim, CLMAConfig.dropout)
        self.mr1_element_projection = Projection(CLMAConfig.mr1_element_model_embedding, CLMAConfig.projection_dim, CLMAConfig.dropout)
        self.mr2_projection = Projection(CLMAConfig.mr2_model_embedding, CLMAConfig.projection_dim, CLMAConfig.dropout)
        self.mr2_element_projection = Projection(CLMAConfig.mr2_model_embedding, CLMAConfig.projection_dim, CLMAConfig.dropout)

        self.alpha = CLMAConfig.alpha

    def forward(self, batch):

        # Get ppm of CNMR
        mr1_input = batch[self.mr1_name]
        
        mr1_element = batch[self.mr1_element_name]

        for key, tensor in mr1_element.items():
            mr1_element[key] = tensor.to(self.device)
        
        element_diff = mr1_element['diff']

        IE_list = mr1_element['num_peaks']
 
        mr2_input = batch[self.mr2_name]

        element_targets = F.softmax(element_diff, dim = -1)
        
        # Getting pre-Embeddings(features) of mr1 and mr2 from Encoders (acceptable with different dimension)
        mr1_features = self.mr1_encoder(mr1_input.to(self.device))
        mr1_element_features = self.mr1_element_encoder(mr1_element)


        mr2_features, mr2_element_features = self.mr2_encoder(mr2_input.to(self.device))
       
        # Getting post-Embeddings of mr1 and mr2 from Projection (mandatory with same dimension)
        mr1_embeddings = self.mr1_projection(mr1_features)
        mr2_embeddings = self.mr2_projection(mr2_features)
        

        mr1_element_embeddings = self.mr1_element_projection(mr1_element_features)
        mr2_element_embeddings = self.mr2_element_projection(mr2_element_features)


        # Computing IE-Loss   
        mr1_mr2_element_logits = mr1_element_embeddings @ mr2_element_embeddings.T

        mr1_element_crossmodal_loss = cross_entropy(mr1_mr2_element_logits, element_targets)
        mr2_element_crossmodal_loss = cross_entropy(mr1_mr2_element_logits.T, element_targets.T)

        loss_element = mr1_element_crossmodal_loss + mr2_element_crossmodal_loss
   
        # Computing RS-Loss
        logits = (mr1_embeddings @ mr2_embeddings.T)

        mr1_similarity = mr1_embeddings @ mr1_embeddings.T
        mr2_similarity = mr2_embeddings @ mr2_embeddings.T

        n = self.batch_size
        targets = torch.eye(n)
        targets = targets.to(self.device)

 
        mr1_loss = cross_entropy(logits, targets)
        mr2_loss = cross_entropy(logits.T, targets.T)
        loss_mr = mr1_loss + mr2_loss
        # overall loss    
        loss = self.alpha * loss_mr + (1-self.alpha) * loss_element 

        return loss, loss_mr, loss_element, logits, mr1_mr2_element_logits, IE_list, mr1_element['peak']



def cross_entropy(preds, targets, reduction='mean'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

