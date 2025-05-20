from spatiotemporal_postprocessing.losses import get_loss
from spatiotemporal_postprocessing.nn import get_model
from spatiotemporal_postprocessing.datasets import get_datamodule
from spatiotemporal_postprocessing.utils import mask_anomalous_targets, log_prediction_plots, plot_rank_histogram, load_checkpoint, save_checkpoint
import spatiotemporal_postprocessing.losses.probabilistic as loss_prob
import spatiotemporal_postprocessing.losses.deterministic as loss_det
import xarray as xr
from tsl.ops.connectivity import adj_to_edge_index
import torch
from torch.utils.data import DataLoader
import mlflow
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import numpy as np
from collections import defaultdict
import optuna
from optuna.samplers import GridSampler
from tqdm import tqdm
import random
from train_multiscalewave import train_multigraphwavenet, validate_multigraphwavenet
from test_multiscalewave import test_graph_wavenet

MASK_ANOM = False # Set to True to Mask Anomalous Targets
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def grid_space():
    # this must have the same space as get_search_space, where we want to sample
    return {
        #"knn" : [5, 10, 15],
        #"theta" : ["median", "factormedian", "std"],
        #"seed" : [3,4,5],
        "lr" : [0.0001,0.0005,0.00002]
    }

def get_search_space(trial):
    space = {
        "hidden_channels": 32,#trial.suggest_categorical("hidden_channels", [16,32,48]),
        "num_layers": 2,#trial.suggest_categorical("num_layers", [2, 3]),
        "dropout_p": 0.2,
        "lr": trial.suggest_categorical("lr", [0.0001,0.0005,0.00002]),#0.0001,#8652217203724423,#trial.suggest_float("lr", 0.00002, 0.0002,),
        "weight_decay": 1e-5,
        "scheduler": "CosineAnnealingWarmRestarts",
        "knn" : 5,#trial.suggest_categorical("knn", [5, 10, 15]),
        "threshold" : 0.6, # 0.9
        "theta" : "std",#trial.suggest_categorical("theta", ["median", "factormedian", "std"]),
        "seed": 0,#trial.suggest_categorical("seed", [3,4,5]),
        #"scheduler": trial.suggest_categorical("scheduler", ["OneCycleLR","CosineAnnealingWarmRestarts","StepLR","ExponentialLR"])
    }

    if space["scheduler"]=="onecycle":
        space["pct_start"] = 0.1
    elif space["scheduler"]=="cosine":
        space["t_max"] = 100
    elif space["scheduler"]=="steplr":
        space["step_size"] = 1
        space["step_gamma"] = 0.5
    else:
        space["exp_gamma"] = 0.9564069612741963
    return space

def update_config_with_trial(cfg: DictConfig, search_space):
    # Update specific parts of the configuration with values from the trial
    if cfg.model.type != "MultiScaleGraphWaveNet":
        cfg.model.kwargs.hidden_channels = search_space['hidden_channels']
        cfg.model.kwargs.num_layers = search_space['num_layers']
        cfg.model.kwargs.dropout_p = search_space['dropout_p']
        cfg.training.optim.kwargs.lr = search_space['lr']
        cfg.training.optim.kwargs.weight_decay = search_space['weight_decay']
        cfg.training.scheduler.algo = search_space['scheduler']
        cfg.graph_kwargs.knn = search_space['knn']
        cfg.graph_kwargs.threshold = search_space['threshold']
        cfg.graph_kwargs.theta = search_space['theta']
        cfg.seed = search_space["seed"]

        # Update scheduler-specific parameters if they exist
        if 'pct_start' in search_space:
            cfg.training.scheduler.kwargs.pct_start = search_space['pct_start']
        if 't_max' in search_space:
            cfg.training.scheduler.kwargs.t_max = search_space['t_max']
        if 'step_size' in search_space:
            cfg.training.scheduler.kwargs.step_size = search_space['step_size']
            cfg.training.scheduler.kwargs.step_gamma = search_space['step_gamma']
            cfg.training.scheduler.kwargs.gamma = search_space['step_gamma']
        if 'exp_gamma' in search_space:
            cfg.training.scheduler.kwargs.exp_gamma = search_space['exp_gamma']

    return cfg


# NOTE uncomment to debug issues related to autograd
#torch.autograd.set_detect_anomaly(True)

OmegaConf.register_new_resolver("add_one", lambda x: int(x) + 1)
@hydra.main(version_base="1.1", config_path="./configs", config_name="default_training_conf")
def app(cfg: DictConfig) -> float:
    global TRIAL
    trial = TRIAL
    print("cfg2", cfg)

    if OmegaConf.select(cfg, "training.optim.kwargs.betas") is not None:
        cfg.training.optim.kwargs.betas = eval(cfg.training.optim.kwargs.betas)
    #if 'hidden_channels' in cfg.model.kwargs:
    #    cfg.model.kwargs.hidden_channels = eval(cfg.model.kwargs.hidden_channels) 
    
    search_space = get_search_space(trial)
    cfg = update_config_with_trial(cfg, search_space)
    print(cfg)
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(cfg.seed)
    ds = xr.open_dataset(cfg.dataset.features_pth)
    ds_targets = xr.open_dataset(cfg.dataset.targets_pth)

    dm = get_datamodule(ds=ds, ds_targets=ds_targets,
                        val_split=cfg.dataset.val_split,
                        test_start_date=cfg.dataset.test_start,
                        train_val_end_date=cfg.dataset.train_val_end,
                        lead_time_hours=cfg.dataset.hours_leadtime,
                       predictors=cfg.dataset.predictors, target_var=cfg.dataset.target_var,
                       return_graph=True, graph_kwargs=cfg.graph_kwargs)
    print(dm)

    adj_matrix = dm.adj_matrix
    edge_index, edge_weight = adj_to_edge_index(adj=torch.tensor(adj_matrix)) # NOTE not using w_ij for now
    
    
    batch_size = cfg.training.batch_size
    train_dataloader = DataLoader(dm.train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dm.val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dm.test_dataset, batch_size=batch_size, shuffle=True)
    
    assert dm.train_dataset.stations == dm.val_dataset.stations == dm.test_dataset.stations # sanity check 

    if cfg.model.type != "MultiScaleGraphWaveNet":
        model_kwargs = {'input_size': dm.train_dataset.f, 
                        'n_stations': dm.train_dataset.stations,
                        **cfg.model.kwargs} 
    else:
        model_kwargs = {'n_stations': dm.train_dataset.stations,
                        "in_channels": len(cfg.dataset.predictors),
                        "history_len": cfg.d_map.ch2,
                        "adj_matrix": adj_matrix,
                        **cfg.model.kwargs} 

    model = get_model(model_type=cfg.model.type, **model_kwargs)

    # model = model = MultiScaleGraphWaveNet(
    #    num_nodes=N,
    #    in_channels=P,
    #    history_len=L, 
    #    adj_matrix=dm.adj_matrix, 
    #).to(device)
    
    epochs = cfg.training.epochs
    
    criterion = loss_prob.MaskedCRPSLogNormal() #get_loss(cfg.training.loss)
    l1_crit = loss_det.MaskedL1Loss()# MaskedMAE()
    #crps_crit  = MaskedCRPSLogNormal()
    #l1_crit    = MaskedL1Loss()
    optimizer = getattr(torch.optim, cfg.training.optim.algo)(model.parameters(), **cfg.training.optim.kwargs)

    USE_PRETRAINED = False#"./checkpoints/model_GRUGCNModel_2_epoch_95.pt" # checkpoint path
    if USE_PRETRAINED:
        _, _, start_epoch = load_checkpoint(model, optimizer, USE_PRETRAINED)

    def filter_scheduler_kwargs(scheduler_cls, kwargs):
        import inspect
        # Get the parameter names for the scheduler's __init__ method
        valid_params = inspect.signature(scheduler_cls.__init__).parameters
        # Filter kwargs to include only valid parameters
        return {k: v for k, v in kwargs.items() if k in valid_params}

    scheduler_cls = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.algo)
    filtered_kwargs = filter_scheduler_kwargs(scheduler_cls, cfg.training.scheduler.kwargs)
    lr_scheduler = scheduler_cls(optimizer, **filtered_kwargs)

    gradient_clip_value = cfg.training.gradient_clip_value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    edge_index = edge_index.to(device)
    print("device", device)
    
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.logging.experiment_id)

    if device_type == 'cpu':
        torch.set_num_threads(16)
    with mlflow.start_run():
        mlflow.log_param("device_type", device_type)
        mlflow.log_param("optimizer", type(optimizer).__name__) 
        mlflow.log_param("criterion", type(criterion).__name__) 
        mlflow.log_param("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(cfg_dict, 'training_config.json')
        mlflow.log_params(cfg_dict)
        
        print('Training started. Check your logs on MLFlow.')

        lowest_val_loss = float("inf")

        if cfg.model.type == "MultiScaleGraphWaveNet":
            for epoch in range(epochs):
                train_multigraphwavenet(model, train_dataloader, optimizer, lr_scheduler, device, epochs)
                avg_val_loss, avg_val_acc, avg_crps, avg_mae = validate_multigraphwavenet(model, val_dataloader, cfg, device)
                mlflow.log_metric("Val CRSP", avg_val_loss, step=epoch)
                mlflow.log_metric("Val MAE", avg_val_acc, step=epoch)
                for h in [1, 24, 48, 96]:
                    print(f"{h:8d} │ {avg_crps[h]:8.4f} │ {avg_mae[h]:8.4f}")
                    mlflow.log_metric(f"crps h {h}", avg_crps[h], step=epoch)
                    mlflow.log_metric(f"mae h {h}", avg_mae[h], step=epoch)
        else:

        
            for epoch in range(epochs):
                # ——— TRAIN ————————————————————————————————————————————————
                model.train()
                train_crps_sum = 0.0
                train_mae_sum  = 0.0

                for i, (x_batch, y_batch) in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch} Train"):
                    x = x_batch.to(device)   # [B,L,N,P]
                    if MASK_ANOM: y_batch = mask_anomalous_targets(y_batch, min_speed=0.2, max_speed=10.0)
                    y = y_batch.to(device)   # [B,L,N,1]

                    optimizer.zero_grad()
                    dist = model(x, edge_index)          # LogNormal distribution over [B,L,N]

                    # 1) CRPS loss
                    loss = criterion(dist, y)

                    # 2) MAE on the predictive mean
                    mu    = dist.mean.squeeze(-1)  # [B,L,N]
                    truth = y.squeeze(-1)          # [B,L,N]
                    mae   = l1_crit(mu, truth)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step(epoch + i/len(train_dataloader))

                    train_crps_sum += loss.item()
                    train_mae_sum  += mae.item()

                avg_train_crps = train_crps_sum / len(train_dataloader)
                avg_train_mae  = train_mae_sum  / len(train_dataloader)

                lr_scheduler.step(epoch=epoch) # TODO make this generic (not all lr_sched use the epoch as param)
                for group, lr in enumerate(lr_scheduler.get_last_lr()):
                    mlflow.log_metric(f'lr_{group}', lr, step=epoch)

                # ——— VALIDATION ——————————————————————————————————————
                model.eval()
                val_crps_sum = 0.0
                val_mae_sum  = 0.0

                # prepare per-horizon accumulators
                horizons    = [1, 24, 48, 96]
                sum_crps    = {h: 0.0 for h in horizons}
                count_crps  = {h: 0   for h in horizons}
                sum_mae     = {h: 0.0 for h in horizons}
                count_mae   = {h: 0   for h in horizons}

                with torch.no_grad():
                    firstbatch = True
                    plot_rank_histogram(model, val_dataloader, edge_index, model_name=cfg.model.type)
                    for x_batch, y_batch in tqdm(val_dataloader, desc=f"Epoch {epoch} Valid"):
                        x = x_batch.to(device)
                        if MASK_ANOM: y_batch = mask_anomalous_targets(y_batch, min_speed=0.2, max_speed=10.0)
                        y = y_batch.to(device)

                        if cfg.model.type != "MultiScaleGraphWaveNet":
                            dist = model(x, edge_index)
                        else:
                            dist = model(x)
                        loss = criterion(dist, y)

                        mu    = dist.mean.squeeze(-1)  # [B,L,N]
                        truth = y.squeeze(-1)          # [B,L,N]
                        mae   = l1_crit(mu, truth)

                        val_crps_sum += loss.item()
                        val_mae_sum  += mae.item()

                        # per-horizon metrics
                        for h in horizons:
                            # slice out lead‐time h (0-based index!)
                            mu_h    = mu[:, h, :]       # [B,N]
                            y_h     = truth[:, h, :]    # [B,N]
                            valid   = (~torch.isnan(y_h)).sum().item()

                            # CRPS at horizon h (no extra unsqueeze)
                            loc_h   = dist.loc[:, h, :, :].squeeze(-1)   # [B,N]
                            scale_h = dist.scale[:, h, :, :].squeeze(-1) # [B,N]
                            d_h     = torch.distributions.LogNormal(loc_h, scale_h)
                            c_h     = criterion(d_h, y_h).item()

                            sum_crps[h]   += c_h * valid
                            count_crps[h] += valid

                            # MAE at horizon h
                            m_h = l1_crit(mu_h, y_h).item()
                            sum_mae[h]   += m_h * valid
                            count_mae[h] += valid

                        if firstbatch:
                            firstbatch = False # Only plot for one batch
                            log_prediction_plots(x=x_batch, 
                                            y=y_batch, 
                                            pred_dist=dist, 
                                            example_indices=[0,0,0,0], 
                                            stations=[1,2,3,4],
                                            epoch=epoch,
                                            input_denormalizer=dm.val_dataset.input_denormalizer,
                                            model_name=cfg.model.type)

                avg_val_crps = val_crps_sum / len(val_dataloader)
                avg_val_mae  = val_mae_sum  / len(val_dataloader)
                avg_crps_h   = {h: sum_crps[h]/count_crps[h] for h in horizons}
                avg_mae_h    = {h: sum_mae[h]/count_mae[h]   for h in horizons}

                print(avg_val_crps)
                print(avg_val_mae)
                print(avg_crps_h)
                print(avg_mae_h)

                if avg_val_crps < lowest_val_loss: 
                    lowest_val_loss = avg_val_crps

                print(f"\n=== Validation Metrics @ Epoch {epoch+1} ===")
                print(f"Train CRPS ", avg_train_crps)
                mlflow.log_metric("Train CRPS", avg_train_crps, step=epoch)
                print(f"Train MAE ", avg_train_mae)
                mlflow.log_metric("Train MAE", avg_train_mae, step=epoch)
                print(f"Validation CRPS ", avg_val_crps)
                mlflow.log_metric("Validation CRPS", avg_val_crps, step=epoch)
                print(f"Validation MAE ", avg_val_mae)
                mlflow.log_metric("Validation MAE", avg_val_mae, step=epoch)
                print(f"{'Lead(h)':>8} │ {'CRPS':>8} │ {'MAE':>8}")
                print("-" * 30)
                for h in [1, 24, 48, 96]:
                    print(f"{h:8d} │ {avg_crps_h[h]:8.4f} │ {avg_mae_h[h]:8.4f}")
                    mlflow.log_metric(f"crps h {h}", avg_crps_h[h], step=epoch)
                    mlflow.log_metric(f"mae h {h}", avg_mae_h[h], step=epoch)
                save_checkpoint(epoch, model, optimizer, checkpoint_dir, name=cfg.model.type)

        if cfg.model.type == "MultiScaleGraphWaveNet":
            test_graph_wavenet(model, cfg.dataset.features_pth, cfg.dataset.targets_pth, cfg.model.kwargs, cfg.graph_kwargs, cfg.model.type)
        else:
            evaluate(cfg, model) # This runs the Test Set

        return lowest_val_loss

# ——— TEST ——————————————————————————————————————
def evaluate(cfg: DictConfig, model):
    # Load datasets
    ds = xr.open_dataset(cfg.dataset.features_pth)
    ds_targets = xr.open_dataset(cfg.dataset.targets_pth)
    dm = get_datamodule(ds=ds, ds_targets=ds_targets,
                        val_split=cfg.dataset.val_split,
                        test_start_date=cfg.dataset.test_start,
                        train_val_end_date=cfg.dataset.train_val_end,
                        lead_time_hours=cfg.dataset.hours_leadtime,
                        predictors=cfg.dataset.predictors, 
                        target_var=cfg.dataset.target_var,
                        return_graph=True,
                        graph_kwargs=cfg.graph_kwargs)


    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataloader = DataLoader(dm.test_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    adj_matrix = dm.adj_matrix
    edge_index, edge_weight = adj_to_edge_index(adj=torch.tensor(adj_matrix)) # NOTE not using w_ij for now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_index = edge_index.to(device)

    model.eval()

    # STATISTICS
    def test_model(model, loader, device):
        means, stds = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                # x_batch: (B, C, N, T)
                x_batch = x_batch.to(device)
                #x_batch = x_batch.permute(0, 3, 2, 1)
                if cfg.model.type != "MultiScaleGraphWaveNet":
                    dist = model(x_batch, edge_index)
                else:
                    dist = model(x_batch)
                means.append(dist.loc.cpu().numpy())
                stds.append(dist.scale.cpu().numpy())
        return np.concatenate(means, axis=0), np.concatenate(stds, axis=0)
    
    # Load, test, and save outputs
    mean_vals, std_vals = test_model(model, test_dataloader, device)
    
    # Save distributions over all test samples
    os.makedirs('outputs', exist_ok=True)
    output_file = os.path.join('outputs', f'test_outputs_{cfg.model.type}_seed_{cfg.seed}.npz')
    np.savez_compressed(output_file, mean=mean_vals, std=std_vals)
    mlflow.log_artifact(output_file)
    print(f"Saved test outputs to {output_file}")

    # Prepare evaluation metrics
    l1_crit = loss_det.MaskedL1Loss()# MaskedMAE()
    criterion = loss_prob.MaskedCRPSLogNormal() 

    with torch.no_grad():
        test_mae_sum = 0.0
        test_crps_sum = 0.0

        # prepare per-horizon accumulators
        horizons    = [1, 24, 48, 96]
        sum_crps    = {h: 0.0 for h in horizons}
        count_crps  = {h: 0   for h in horizons}
        sum_mae     = {h: 0.0 for h in horizons}
        count_mae   = {h: 0   for h in horizons}

        for i, (x_batch, y_batch) in enumerate(test_dataloader):
            print(f"testing batch {i} of {len(test_dataloader)}")
            x = x_batch.to('cuda')
            y = y_batch.to('cuda')
            dist = model(x, edge_index)
            mu = dist.mean.squeeze(-1)  # [B,L,N]
            truth = y.squeeze(-1)  # [B,L,N]
            loss_crps = criterion(dist, y)
            mae = l1_crit(mu, truth)
            test_mae_sum += mae.item()
            test_crps_sum += loss_crps.item()

            # per-horizon metrics
            for h in horizons:
                # slice out lead‐time h (0-based index!)
                mu_h    = mu[:, h, :]       # [B,N]
                y_h     = truth[:, h, :]    # [B,N]
                valid   = (~torch.isnan(y_h)).sum().item()

                # CRPS at horizon h (no extra unsqueeze)
                loc_h   = dist.loc[:, h, :, :].squeeze(-1)   # [B,N]
                scale_h = dist.scale[:, h, :, :].squeeze(-1) # [B,N]
                d_h     = torch.distributions.LogNormal(loc_h, scale_h)
                c_h     = criterion(d_h, y_h).item()

                sum_crps[h]   += c_h * valid
                count_crps[h] += valid

                # MAE at horizon h
                m_h = l1_crit(mu_h, y_h).item()
                sum_mae[h]   += m_h * valid
                count_mae[h] += valid

            if i == 0:
                log_prediction_plots(x=x_batch, y=y_batch, pred_dist=dist,
                                    example_indices=[0, 0, 0, 0], stations=[1, 2, 3, 4],
                                    epoch=0, input_denormalizer=dm.test_dataset.input_denormalizer,
                                    model_name=cfg.model.type)
        
        avg_crps_h   = {h: sum_crps[h]/count_crps[h] for h in horizons}
        avg_mae_h    = {h: sum_mae[h]/count_mae[h]   for h in horizons}
        for h in [1, 24, 48, 96]:
            print(f"{h:8d} │ {avg_crps_h[h]:8.4f} │ {avg_mae_h[h]:8.4f}")
            mlflow.log_metric(f"test crps h {h}", avg_crps_h[h], step=0)
            mlflow.log_metric(f"test mae h {h}", avg_mae_h[h], step=0)
        plot_rank_histogram(model, test_dataloader, edge_index, model_name=cfg.model.type)
        avg_test_crps = test_crps_sum / len(test_dataloader)
        avg_test_mae = test_mae_sum / len(test_dataloader)
        print(f"Test MAE: {avg_test_mae}")
        mlflow.log_metric("Test MAE", avg_test_mae)
        print(f"Test CRPS: {avg_test_crps}")
        mlflow.log_metric("Test CRPS", avg_test_crps)
        return avg_test_crps



@hydra.main(version_base="1.1", config_path="./configs", config_name="default_training_conf")
def set_trial(trial):
    # this function is needed to update the TRIAL, because we can not pass it as an argument to app()
    # because of the cfg
    global TRIAL
    TRIAL = trial
    value = app()
    print(f"Trial {trial.number}: {trial.params}, Value: {value}")

    return value

if __name__ == '__main__':
    # Define a study
    study_name = "Enhanced_BiDirectionalSTGNN"
    study = optuna.create_study(study_name=study_name, direction="minimize", sampler=GridSampler(grid_space()))

    # Optimize the study with the objective function
    study.optimize(set_trial, n_trials=5)
    
    # After optimization, you can retrieve the best result
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.params}")