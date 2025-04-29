import pandas as pd
from Utils.CLMAConfig import CLMAConfig
from Utils.BuildDatasetLoader_5fold import build_dataset_loader
from GraphModels.NG import GNNNodeEncoder, GNNGraphEncoder
from CLMAModels.CLMAModel import CLMAModel
import torch
from SequenceModels.CNMRModel import CNMRModel
from SequenceModels.PeakEncoder import PeakEncoder

from Utils.TrainEpoch import train_epoch
from Utils.ValidEpoch import valid_epoch
import os
import argparse
# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='gin', help="Model type.")
parser.add_argument("--num_layer", type=int, default=5, help="Number of layers.")
parser.add_argument("--embed_dim", type=int, default=128, help="Embed dimension.")
parser.add_argument("--n", type=int, default=5, help="mr1_element_temperature.")
parser.add_argument("--p", type=int, default=1, help="mr1_element_diff_temperature.")
parser.add_argument("--a", type=float, default=0.1, help="alpha for mr_loss")
parser.add_argument("--f", type=int, default=0, help="fold")

# Parse and print the results
args = parser.parse_args()
out_name = "best" + "_n" + str(args.n) + "_p" + str(args.p) + "_" + args.type + "_" + str(args.num_layer) + "_" + str(args.embed_dim) + "_" + str(args.a) + "_" + str(args.f)
print('Output name:%s' % out_name)

def main():
    # Load graph and image file names into a dataframe
    CLMAConfig.mr1_element_temperature = 10 ** (-args.n)
    CLMAConfig.mr1_element_diff_temperature = 10 ** (args.p)
    CLMAConfig.alpha = args.a

    file_name = "cnmr_match_use_2W.csv"
    df = pd.read_csv(os.path.join(CLMAConfig.dataset_path, file_name))

   # Generate train_dataset_loader and valid_dataset_loader
    train_dataset_loader, valid_dataset_loader = build_dataset_loader(df, CLMAConfig)
    CLMAConfig.mr2_model_embedding = args.embed_dim
    CLMAConfig.mr1_element_model_embedding = args.embed_dim
    
    nodeEncoder = GNNNodeEncoder(args.num_layer, args.embed_dim, JK="last", gnn_type=args.type, aggr='add').to(CLMAConfig.device)
    graph_model = GNNGraphEncoder(nodeEncoder, args.embed_dim, graph_pooling="add").to(CLMAConfig.device)
    ### pre-trained CNMR model
    cnmr_model = CNMRModel()
    state_dict = torch.load('SequenceModels/PretrainedWeights/nmr_model_weights.pth')
    cnmr_model.load_state_dict(state_dict)

    ##PeakEncoder
    peak_model = PeakEncoder(emb_dim=CLMAConfig.mr1_element_model_embedding)
    device = CLMAConfig.device
    model = CLMAModel(cnmr_model, peak_model, graph_model, CLMAConfig).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CLMAConfig.lr, weight_decay=CLMAConfig.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CLMAConfig.patience, factor=CLMAConfig.factor
    )

    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CLMAConfig.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_dataset_loader[args.f], optimizer, lr_scheduler, step, CLMAConfig.accuracies_req)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_dataset_loader[args.f], CLMAConfig.accuracies_req )

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), out_name + ".pt")
            print("Saved Best Model!")

        print("\n")



if __name__ == "__main__":
    main()



