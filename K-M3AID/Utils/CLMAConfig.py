import torch

class CLMAConfig:
    debug = True
    project_path = "./"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    # Parameters for Build Dataset Loader
    batch_size = 128
    validation_ratio = 0.2
    random_seed = 42
    shuffle = True
    drop_last = True # It has to be true
    dataset_path = "./Datasets/CNMR_Dataset/"
    #mr1
    cnmr_path = "./Datasets/CNMR_Dataset/alignment/"
    #mr2
    graphs_path = "./Datasets/CNMR_Dataset/graph_hyb/"

    # Parameters for training
    lr = 1e-3
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 200

    # alpha * mr_loss + (1-alpha) * element_loss
    alpha = 0.1

    # Parameters for Molecular Representation 1 Model
    mr1_name = "cnmr"
    mr1_model_name = 'cnmrmodel'
    mr1_element_name = 'peak'
    mr1_model_embedding = 768
    mr1_model_pretrained = True 
    mr1_model_trainable = False #fine-tunning

    mr1_model_pretrained = False
    mr1_element_model_trainable=True    
    mr1_element_model_embedding = 128
    mr1_element_temperature = 1e-5 
    mr1_element_diff_temperature = 1e1

    # Parameters for Molecular Representation 2 Model
    mr2_name = "graph"
    mr2_model_name = "gin"
    mr2_model_embedding = 128
    mr2_model_pretrained = False
    mr2_model_trainable = True

    # Parameters for Projection
    num_projection_layers = 1
    projection_dim = [512]
    dropout = 0.1

    # Paramerts for CMRP Model
    temperature = 1.0

    #top accuracies
    accuracies_req = [1]
