import matplotlib.pyplot as plt 
import mlflow
import numpy as np
import torch 
import os

def mask_anomalous_targets(y, min_speed, max_speed):
    squeezed = (y.squeeze(-1) if y.dim()==4 else y)
    bad = (squeezed < min_speed) | (squeezed > max_speed) | torch.isnan(squeezed)
    y_clean = squeezed.clone()
    y_clean[bad] = float('nan')
    return y_clean.unsqueeze(-1) if y.dim()==4 else y_clean

def log_prediction_plots(x, y, pred_dist, example_indices, stations, epoch, input_denormalizer, model_name=""):
    x = input_denormalizer(x) # bring inputs to their original range
    x = x.detach().cpu().numpy() 
    y = y.detach().cpu().numpy()
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))  
    axs = axs.flatten() 
    
    quantile_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).repeat(*y.shape).to(pred_dist.mean.device)
    quantiles = pred_dist.icdf(quantile_levels).detach().cpu().numpy()
    #quantiles = np.swapaxes(quantiles, 1, 2)
    #print(quantiles)
    
    time = time = np.arange(x.shape[1])
     
    for i, (b_idx, station) in enumerate(zip(example_indices, stations)):
        ax = axs[i]
        # First feature is expected to be the ensemble mean
        ax.plot(x[b_idx, :, station, 0], label='ens_mean', color='forestgreen')
       
        # Plot quantiles 
        ax.fill_between(time, quantiles[b_idx,:, station,0], quantiles[b_idx,:, station,1], 
                    alpha=0.15, color="blue", label="5%-95%")
        
        ax.fill_between(time, quantiles[b_idx,:, station,1], quantiles[b_idx,:, station,2], 
                        alpha=0.35, color="blue", label="25%-75%")
        
        ax.plot(time, quantiles[b_idx, :, station,2], color="black", linestyle="--", label="Median (50%)")
        
        ax.fill_between(time, quantiles[b_idx,:, station,2], quantiles[b_idx,:, station,3], 
                        alpha=0.35, color="blue")
        
        ax.fill_between(time, quantiles[b_idx,:, station,3], quantiles[b_idx,:, station,4], 
                        alpha=0.15, color="blue")
        
        ax.plot(y[b_idx, :, station, 0], label='observed', color='mediumvioletred')
        ax.set_title(f'Station {station} at batch element {b_idx}')
        ax.set_xlabel("Lead time")
        ax.set_ylabel("Wind speed")
        
    axs[-1].legend() # only show legend in the last plot

    plt.suptitle(f'Predictions at Epoch {epoch} for model {model_name}')
    plt.tight_layout()


    plot_filename = f"{model_name}_predictions_epoch_{epoch}.png"
    plt.savefig(plot_filename)
    plt.close(fig) 

    mlflow.log_artifact(plot_filename)



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def plot_rank_histogram(
    model,
    dataloader: DataLoader,
    edge_index,
    dm=None,
    model_name: str = "",
    n_samples: int = 20,
    horizons: list = [1,24,48,96],
    plot_dir: str = ".",
):
    """
    model       -- your trained model (already .to(device) and with state_dict loaded)
    dataloader  -- e.g. DataLoader(dm.val_dataset, batch_size=32, shuffle=False)
    edge_index  -- from adj_to_edge_index(dm.adj_matrix)
    dm          -- your datamodule (for denormalizer if needed)
    model_name  -- one of "baseline","tcn_gnn","bidirectionalstgnn"
    n_samples   -- number of trajectories to sample (20)
    horizons    -- list of lead‐time indices to compute histograms for
    """
    device = next(model.parameters()).device
    model.eval()
    edge_index = edge_index.to(device)

    # collect ranks per horizon
    ranks = {h: [] for h in horizons}

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)                                  # [B, L, N,  F]
            y = y.to(device).squeeze(-1)                      # [B, L, N]
            y = mask_anomalous_targets(y, min_speed=0.2, max_speed=10.0)

            # get distribution
            dist = model(x, edge_index) if hasattr(model, "forward") and model_name!="baseline" else model(x)
            # sample n_samples trajectories: returns [n_samples, B, L, N]
            samp = dist.rsample((n_samples,)).squeeze(-1).cpu().numpy()
            truth = y.cpu().numpy()                            # [B, L, N]

            # for each horizon, compute ranks
            for h in horizons:
                # samp[:, b, h, n] < truth[b,h,n] -> sum over axis=0 gives rank in [0..n_samples]
                below = (samp[:, :, horizons.index(h), :] < truth[:, horizons.index(h), :][None,:,:])
                # but careful: horizons.index(h) gives position; we want direct h-th lead
                # → better to index by h directly:
                s_h = samp[:, :, h, :]    # [n_samples, B, N]
                t_h = truth[:, h, :]      # [B, N]
                # rank = number of samples below truth
                r_h = np.sum(s_h < t_h[None,:,:], axis=0)  # shape [B, N]
                ranks[h].extend(r_h.flatten().tolist())

    # now plot 2x2 histograms
    plot_dir = plot_dir +f"/{model_name}"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axs = plt.subplots(2,2,figsize=(16,10))
    axs = axs.flatten()
    for i, h in enumerate(horizons):
        axs[i].hist(ranks[h], bins=np.arange(n_samples+2)-0.5, edgecolor="k")
        axs[i].set_title(f"Rank histogram — lead {h}h")
        axs[i].set_ylabel("Frequency")
        axs[i].set_xlim(-0.5, n_samples+0.5)

    plt.tight_layout()
    outpath = os.path.join(plot_dir, "rankhist_all.png")
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    mlflow.log_artifact(outpath)
    print(f"Saved rank histograms to {outpath}")