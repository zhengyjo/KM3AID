from DatasetModels.CLMADataset import CLMADataset
from DatasetModels.GraphDataset import GraphDataset
from DatasetModels.CNMRDataset import CNMRDataset
from DatasetModels.PeakDataset import PeakDataset

import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pandas as pd


def build_dataset_loader(dataframe, CLMAConfig):

    # Load pars from CMRPConfig
    mr1_name = CLMAConfig.mr1_name
    mr1_element_name = CLMAConfig.mr1_element_name
    mr2_name = CLMAConfig.mr2_name
    random_seed = CLMAConfig.random_seed
    validation_ratio = CLMAConfig.validation_ratio
    batch_size = CLMAConfig.batch_size
    shuffle = CLMAConfig.shuffle
    drop_last = CLMAConfig.drop_last

    # Create mr1_dataset and mr2_dataset instances
    dataset_mapping = {
        'graph': GraphDataset,
        'cnmr': CNMRDataset,
        'peak': PeakDataset 
    }

    mr1_dataset = dataset_mapping.get(mr1_name)(dataframe[mr1_name], CLMAConfig)
    mr1_element_dataset = dataset_mapping.get(mr1_element_name)(dataframe[mr1_name], CLMAConfig)
    mr2_dataset = dataset_mapping.get(mr2_name)(dataframe[mr2_name], CLMAConfig)

    # Create CGIPDataset instance
    clma_dataset = CLMADataset(mr1_dataset, mr1_element_dataset, mr2_dataset, CLMAConfig)

    # Get the total number of samples in the dataset
    total_samples = len(clma_dataset)

    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Generate random indices for splitting the dataset
    indices = np.random.permutation(total_samples)

    # Calculate the number of samples for validation
    num_valid_samples = int(total_samples * validation_ratio)

    # Split the indices into training and validation sets
    valid_indices = indices[:num_valid_samples]
    train_indices = indices[num_valid_samples:]

    #df_valid = pd.DataFrame({'valid': valid_indices})
    #df_valid.to_csv('valid_indices.csv', index=False) 
    #df_train = pd.DataFrame({'train': train_indices})
    #df_train.to_csv('train_indices.csv', index=False)   

    # Create the training and validation datasets using the selected indices
    train_dataset = Subset(clma_dataset, train_indices)
    valid_dataset = Subset(clma_dataset, valid_indices)

    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=clma_dataset.collate_fn)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=clma_dataset.collate_fn)
    return train_dataset_loader, valid_dataset_loader

