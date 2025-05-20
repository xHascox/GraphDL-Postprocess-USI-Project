import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
from spatiotemporal_postprocessing.nn.models import MultiScaleGraphWaveNet
from spatiotemporal_postprocessing.losses.probabilistic import MaskedCRPSLogNormalGraphWavenet
from spatiotemporal_postprocessing.losses.deterministic import MaskedMAEGraphWavenet
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from spatiotemporal_postprocessing.datasets import get_datamodule_anomalous
import mlflow



def make_rank_histograms_graph_wavenet(pred_dist: torch.distributions.Distribution,
                         y_true     : torch.Tensor,
                         lead_ids   : list[int],
                         n_samples  : int = 20,
                         epoch      : int | str = "final",
                        root_dir         : str  = "histograms"):
    """
    pred_dist.mean/scale shape : (B , N , T [, C])
    y_true                     : (B , T , N , 1)
    lead_ids                   : list of lead-time indices (e.g. [1,24,48,96])
    """

    y_true = y_true.to(pred_dist.loc.device)

    # --- 1. sample & reshape ------------------------------------------------
    # samples  -> (S , B , N , T)
    samples = pred_dist.rsample((n_samples,)).squeeze(-1)     # remove C / 1-axis
    samples = samples.permute(0, 1, 3, 2)                     # (S,B,T,N)

    # observation -> (B , T , N)
    obs = y_true.squeeze(-1)                                  # (B,T,N)

    # --- 2. iterate over requested lead times ------------------------------
    for t in lead_ids:
        # extract slice  ->  obs_t  (B,N) ;  samp_t  (S,B,N)
        obs_t   = obs[:,  t, :]                # (B,N)
        samp_t  = samples[:, :, t, :]          # (S,B,N)

        # flatten station & batch dims
        obs_1d  = obs_t.flatten()              # (B*N,)
        samp_2d = samp_t.reshape(n_samples, -1)  # (S, B*N)

        # remove NaNs (mask along last dim)
        mask       = torch.isfinite(obs_1d)
        obs_1d     = obs_1d[mask]                # (M,)
        samp_2d    = samp_2d[:, mask]            # (S,M)

        # --- 3. rank calculation ------------------------------------------
        # ranks are counted 1 … (S+1);  histogram has S+1 bins
        # rank = number of samples < obs   (+1)
        ranks = (samp_2d < obs_1d).sum(dim=0).cpu().numpy()   # (M,)
        n_bins = n_samples + 1
        hist   = np.bincount(ranks, minlength=n_bins)

        # --- 4. plotting ---------------------------------------------------
        plt.figure(figsize=(6,4))
        plt.bar(np.arange(n_bins), hist, width=0.9, color='steelblue')
        plt.xlabel("Rank (0 … 20)")
        plt.ylabel("Count")
        plt.title(f"Rank histogram  –  t = {t}h ")
        plt.tight_layout()

        #fname = f"rank_hist_t{t}_ep{epoch}.png"
        #plt.savefig(fname); plt.close()
        #print(f"Logged rank-histogram {fname}")

        # -- folder & save ---------------------------------------------------
        out_dir = os.path.join(root_dir)
        os.makedirs(out_dir, exist_ok=True)
        fname   = os.path.join(out_dir, f"rank_hist_t{t}.png")
        plt.savefig(fname); plt.close()
        mlflow.log_artifact(fname)
        print(f"Saved histogram → {fname}")


def log_prediction_plots_graph_wavenet(x, y, pred_dist,
                         example_indices, stations,
                         epoch, input_denormalizer,
                          root_dir: str = "predictions"):


    #------------------------------------------------------------
    # 1. denormalise inputs, move to numpy
    #------------------------------------------------------------
    x = input_denormalizer(x)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    B, T, N, _ = y.shape

    #------------------------------------------------------------
    # 2. predictive mean  -> (B,T,N,1)
    #------------------------------------------------------------
    mean = pred_dist.mean.squeeze(-1)        # drop channel dim if 1
    mean = mean.permute(0, 2, 1).unsqueeze(-1)   # (B,T,N,1)

    #------------------------------------------------------------
    # 3. five predictive quantiles  -> (B,T,N,5)
    #------------------------------------------------------------
    probs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                         device=pred_dist.loc.device)
    q_list = [pred_dist.icdf(p).unsqueeze(-1) for p in probs]
    quant  = torch.cat(q_list, dim=-1).squeeze(-2)   # (B,N,T,5)
    quant  = quant.permute(0, 2, 1, 3).detach().cpu().numpy() # (B,T,N,5)

    #------------------------------------------------------------
    # 4. plotting loop
    #------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 8)); axs = axs.flatten()
    time = np.arange(T)

    for ax, b_idx, st in zip(axs, example_indices, stations):

        ax.plot(time, x[b_idx, :, st, 0], label='ens_mean', color='forestgreen')
        ax.fill_between(time, quant[b_idx,:,st,0], quant[b_idx,:,st,4],
                        alpha=0.15, color='steelblue', label='5–95 %')
        ax.fill_between(time, quant[b_idx,:,st,1], quant[b_idx,:,st,3],
                        alpha=0.35, color='steelblue', label='25–75 %')
        ax.plot(time, quant[b_idx,:,st,2], ls='--', color='black', label='median')
        ax.plot(time, y[b_idx,:,st,0], color='crimson', label='observed')

        ax.set_title(f'Station {st}   batch {b_idx}')
        ax.set_xlabel('Lead time'); ax.set_ylabel('Wind speed')

    axs[-1].legend(loc='upper left')
    plt.suptitle(f'Predictions – epoch {epoch}')
    plt.tight_layout()

        # -- ensure folder & save -----------------------------------------------------
    os.makedirs(root_dir, exist_ok=True)
    fname = os.path.join(root_dir, f"predictions_epoch_{epoch}.png")
    plt.savefig(fname); plt.close(fig)
    mlflow.log_artifact(fname)
    print(f"Saved fan-chart → {fname}")


# Predefined hyperparameter configuration from grid search
config = {
    'emb_dim': 16,
    'channels': 16,
    'layers': 3,
    'lr': 1e-3,
    'drop': 0.2,
    'edge_dropout': 0.2,
    'dil': (1, 2, 4, 8),
    'kernels': (1, 3, 5, 7),
    'hist_drop': 0.07,
    'history_block': 12,
    'scheduler': 'OneCycleLR',
    'dynamic': [True],
    'node_emb_dim': 20,
    "seed": [0],
    "knn": [5],
    "treshold": 0.6,
    "theta": ["std"],
    "anomalous": False
}

# Global settings for data
data_cfg = {
    'data_path': '/path/to/data',       # adjust to your dataset location
    'batch_size': 64,
    'num_workers': 4,
    'history_len': 96,                  # input sequence length
    'horizon': 96                       # output sequence length
}

# Instantiate data manager and test loader


model_kwargs = {'num_nodes': 152,
                'in_channels': 18,
                'history_len': 96,
                'horizon': 96}

def load_model(path, device, model_kwargs):
    model = MultiScaleGraphWaveNet(**model_kwargs).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def test_model(model, loader, device, dm):
    # Instantiate data manager and test loader
    #criterion = MaskedCRPSLogNormal()
    #mae = MaskedMAE()
    criterion = MaskedCRPSLogNormalGraphWavenet()
    mae = MaskedMAEGraphWavenet()
    means, stds = [], []
    val_loss_sum = 0.0
    val_acc_sum  = 0.0
    sum_crps = {1:0, 24:0, 48:0, 96:0}
    sum_mae  = {1:0, 24:0, 48:0, 96:0}
    count_crps = {1:0, 24:0, 48:0, 96:0}
    count_mae  = {1:0, 24:0, 48:0, 96:0}
    total_iter = 0
    all_dists = []    # will hold one torch.distributions.LogNormal per batch
    all_y     = []    # ground truth for debugging/metrics
    with torch.no_grad():
        for x_batch, y_batch in loader:
            # x_batch: (B, C, N, T)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_gpu = x_batch.permute(0, 3, 2, 1)
            dist = model(x_gpu)
            all_dists.append(dist)
            all_y.append(y_batch) 
            means.append(dist.loc.cpu().numpy())
            stds.append(dist.scale.cpu().numpy())
            val_loss = criterion(dist, y_batch)
            B, L, N, _ = y_batch.shape
            mean = dist.mean
            mean = mean.view(B, N, L, 1)
            mean = mean.permute(0, 2, 1, 3)
            pred_mean = mean.squeeze(-1)     # → [B, L, N]
            true_vals = y_batch.squeeze(-1)
            val_acc, elements   = mae(pred_mean, true_vals)
            # 2.3 accumulate & update bar
            val_loss_sum += val_loss.item()
            val_acc_sum  += val_acc.item()
            for h, idx in zip([1,24,48,96], [1,24,48,96]):
                y_h    = y_batch[:,idx,:,:] # select time step
                y_h    = y_h.unsqueeze(1) # extend shape at second dimension
                crps_mean = criterion(dist, y_h, t=idx).item()
                valid = torch.isfinite(y_h).sum().item() # count valid points
                # 3) accumulate the *sum* of CRPS over all points
                sum_crps[h] += crps_mean * valid
                count_crps[h] += valid
                B, L, N, _ = y_batch.shape
                mean = dist.mean
                mean = mean.view(B, N, L, 1)
                mean = mean.permute(0, 2, 1, 3)
                mean = mean[:, idx, :, :]
                pred_mean = mean.unsqueeze(1)     # → [B, L, N]
                pred_mean = pred_mean.squeeze(-1)
                true_vals = y_h.squeeze(-1)
                error, elements = mae(pred_mean, true_vals)
                error = error.item()     # now a float
                sum_mae[h]  += error * elements
                count_mae[h]    += elements
            if total_iter == 0:
                log_prediction_plots_graph_wavenet(
                    x=x_batch,  # raw inputs
                    y= y_batch,
                    pred_dist=dist,     # or re‐wrap mus/sigmas into distributions
                    example_indices=[0,0,0,0,0],
                    stations=[1,2,3,4],
                    epoch=0,
                    input_denormalizer=dm.test_dataset.input_denormalizer
            )
            total_iter += 1
    val_loss_sum /= len(loader)
    avg_val_acc  = val_acc_sum  / len(loader)
    #scheduler.step(avg_val_loss)
    avg_crps = {h: sum_crps[h]/count_crps[h] for h in sum_crps}
    avg_mae  = {h: sum_mae [h]/count_mae[h] for h in sum_mae }
    avg_crps = {h: float(v) for h, v in avg_crps.items()}
    avg_mae  = {h: float(v) for h, v in avg_mae.items()}

    for h in [1, 24, 48, 96]:
        print(f"{h:8d} │ {avg_crps[h]:8.4f} │ {avg_mae[h]:8.4f}")
        mlflow.log_metric(f"test crps h {h}", avg_crps[h], step=0)
        mlflow.log_metric(f"test mae h {h}", avg_mae[h], step=0)
    print(f"Test MAE: {avg_val_acc}")
    mlflow.log_metric("Test MAE", avg_val_acc)
    print(f"Test CRPS: {val_loss_sum}")
    mlflow.log_metric("Test CRPS", val_loss_sum)

    ys     = torch.cat(all_y,                             dim=0)  # (n_total, T, N, 1)
    return np.concatenate(means, axis=0), np.concatenate(stds, axis=0), val_loss_sum, avg_val_acc, avg_crps, avg_mae, ys

def test_graph_wavenet(model, features_path, targets_path, model_kwargs, graph_kwargs, output_name):
                
    ds = xr.open_dataset(features_path)
    ds_targets = xr.open_dataset(targets_path)

    nwp_model      = "ch2"
    d_map          = {"ch2": 96}          # hours that correspond to each NWP model
    hours_leadtime = d_map[nwp_model]

    val_split            = 0.20
    test_start_date      = "2024-05-16"
    train_val_end_date   = "2023-09-30"
    target_var           = "obs:wind_speed"
    
    predictors = [
        f"{nwp_model}:wind_speed_ensavg",
        f"{nwp_model}:wind_speed_ensstd",
        f"{nwp_model}:mslp_difference_GVE_GUT_ensavg",
        f"{nwp_model}:mslp_difference_BAS_LUG_ensavg",
        "time:sin_hourofday",
        "time:cos_hourofday",
        "time:sin_dayofyear",
        "time:cos_dayofyear",
        "terrain:elevation_50m",
        "terrain:distance_to_alpine_ridge",
        "terrain:tpi_2000m",
        "terrain:std_2000m",
        "terrain:valley_norm_2000m",
        "terrain:sn_derivative_500m",
        "terrain:sn_derivative_2000m",
        "terrain:we_derivative_500m",
        "terrain:we_derivative_2000m",
        "terrain:sn_derivative_100000m",
    ]
    
    dm = get_datamodule_anomalous(
        ds=ds,
        ds_targets=ds_targets,
        val_split=val_split,
        test_start_date=test_start_date,
        train_val_end_date=train_val_end_date,
        lead_time_hours=hours_leadtime,
        predictors=predictors,
        target_var=target_var,
        return_graph=True,
        graph_kwargs=graph_kwargs,
        anomalous=config["anomalous"]
    )

    test_dataloader = DataLoader(dm.test_dataset, batch_size=32, shuffle=False)
                
    # Prepare model instantiation kwargs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
    
    # Load, test, and save outputs
    mean_vals, std_vals, val_loss_sum, avg_val_acc, avg_crps, avg_mae, ys = test_model(model, test_dataloader, device, dm)
    make_rank_histograms_graph_wavenet(
                    pred_dist=torch.distributions.LogNormal(torch.from_numpy(mean_vals).to(device), torch.from_numpy(std_vals).to(device)),
                    y_true=ys,
                    lead_ids=[1,24,48,96],
                    n_samples=20,
                    epoch=0
                )
    
    # Save distributions over all test samples
    os.makedirs('outputs', exist_ok=True)
    output_file = os.path.join('outputs', f'{output_name}_test_outputs.npz')
    np.savez_compressed(output_file, mean=mean_vals, std=std_vals)
    mlflow.log_artifact(output_file)
    # Record the summary metrics
    summary = {
        "run_name": output_name,
        "val_loss": val_loss_sum,
        "avg_val_acc": avg_val_acc,
        "avg_crps": avg_crps,
        "avg_mae": avg_mae
    }
    print(summary)
    # Write it out to a JSON named after the run
    json_path = os.path.join('outputs', f"{output_name}_results.json")
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved test outputs to {output_file}")
