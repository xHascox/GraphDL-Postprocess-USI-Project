import matplotlib.pyplot as plt 
import mlflow
import numpy as np
import torch 

def log_prediction_plots(x, y, pred_dist, example_indices, stations, epoch, input_denormalizer):
    x = input_denormalizer(x) # bring inputs to their original range
    x = x.detach().cpu().numpy() 
    y = y.detach().cpu().numpy()
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))  
    axs = axs.flatten() 
    
    quantile_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).repeat(*y.shape).to(pred_dist.mean.device)
    quantiles = pred_dist.icdf(quantile_levels).detach().cpu().numpy()
    
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

    plt.suptitle(f'Predictions at Epoch {epoch}')
    plt.tight_layout()


    plot_filename = f"predictions_epoch_{epoch}.png"
    plt.savefig(plot_filename)
    plt.close(fig) 

    mlflow.log_artifact(plot_filename)