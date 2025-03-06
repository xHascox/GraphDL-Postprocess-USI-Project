from spatiotemporal_postprocessing.losses import get_loss
from spatiotemporal_postprocessing.nn import get_model
from spatiotemporal_postprocessing.datasets import get_datamodule
import xarray as xr
from tsl.ops.connectivity import adj_to_edge_index
import torch
from torch.utils.data import DataLoader
import mlflow
from omegaconf import DictConfig, OmegaConf
import hydra
from spatiotemporal_postprocessing.utils import log_prediction_plots

# NOTE uncomment to debug issues related to autograd
# torch.autograd.set_detect_anomaly(True)

OmegaConf.register_new_resolver("add_one", lambda x: int(x) + 1)

@hydra.main(version_base="1.1", config_path="./configs", config_name="default_training_conf")
def app(cfg: DictConfig):
    if OmegaConf.select(cfg, "training.optim.kwargs.betas") is not None:
        cfg.training.optim.kwargs.betas = eval(cfg.training.optim.kwargs.betas)
    if 'hidden_sizes' in cfg.model.kwargs:
        cfg.model.kwargs.hidden_sizes = eval(cfg.model.kwargs.hidden_sizes) 
    
    print(cfg)

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
    
    model_kwargs = {'input_size': dm.train_dataset.f, 
                    'n_stations': dm.train_dataset.stations,
                    **cfg.model.kwargs} 
    model = get_model(model_type=cfg.model.type, **model_kwargs)
    
    epochs = cfg.training.epochs
    criterion = get_loss(cfg.training.loss)
    optimizer = getattr(torch.optim, cfg.training.optim.algo)(model.parameters(), **cfg.training.optim.kwargs)
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.algo)(optimizer, **cfg.training.scheduler.kwargs)
    gradient_clip_value = cfg.training.gradient_clip_value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    edge_index = edge_index.to(device)
    
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
        
        total_iter = 0
        print('Training started. Check your logs on MLFlow.')
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()  
                predictions = model(x_batch, edge_index=edge_index)  

                loss = criterion(predictions, y_batch) 
                loss.backward()  
                
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                
                optimizer.step()  
                total_loss += loss.item()

                mlflow.log_metric("loss", loss.item(), step=total_iter)
                
                total_iter += 1

            avg_loss = total_loss / len(train_dataloader)
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            lr_scheduler.step(epoch=epoch) # TODO make this generic (not all lr_sched use the epoch as param)
            for group, lr in enumerate(lr_scheduler.get_last_lr()):
                mlflow.log_metric(f'lr_{group}', lr, step=epoch)
            
            # VALIDATION LOOP
            val_loss = 0
            val_loss_original_range = 0
            model.eval()
            with torch.no_grad():
                tgt_denormalizer = dm.val_dataset.target_denormalizer
                for batch_idx, (x_batch, y_batch) in enumerate(val_dataloader):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = model(x_batch, edge_index=edge_index)  

                    val_loss += criterion(predictions, y_batch).item()
                    val_loss_original_range += criterion(tgt_denormalizer(predictions), tgt_denormalizer(y_batch)).item()
                    
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_loss_or = val_loss_original_range / len(val_dataloader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_loss_original_range", avg_val_loss_or, step=epoch)
            
            # Optional plotting
            if epoch % 10 == 0:
                with torch.no_grad():
                    x_val_batch, y_val_batch = next(iter(val_dataloader))  
                    x_val_batch = x_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
                    val_predictions = model(x_val_batch, edge_index=edge_index)  
                    
                    log_prediction_plots(x=x_val_batch, 
                                        y=y_val_batch, 
                                        pred_dist=val_predictions, 
                                        example_indices=[0,0,0,0], 
                                        stations=[1,2,3,4],
                                        epoch=epoch,
                                        input_denormalizer=dm.val_dataset.input_denormalizer)
            
            
if __name__ == '__main__':
    app()