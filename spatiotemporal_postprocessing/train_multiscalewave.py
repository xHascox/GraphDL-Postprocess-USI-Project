from spatiotemporal_postprocessing.losses.probabilistic import MaskedCRPSLogNormalGraphWavenet
from spatiotemporal_postprocessing.losses.deterministic import MaskedMAEGraphWavenet
import torch
import torch.nn as nn


criterion = MaskedCRPSLogNormalGraphWavenet()
mae = MaskedMAEGraphWavenet()



def train_multigraphwavenet(model, train_loader, optimizer, scheduler, device, total_iter, criterion=criterion, mae=mae):
    """
    Run one training epoch for MSGWN, logging CRPS & MAE to wandb.

    Returns:
        avg_loss (float), avg_error (float), updated total_iter (int)
    """
    model.train()
    train_loss_sum = 0.0
    train_error_sum = 0.0

    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # permute to (B, F, N, T)
        x_in = x_batch.permute(0, 3, 2, 1)
        preds = model(x_in)

        # CRPS loss
        loss = criterion(preds, y_batch)
        
        # MAE on predictive mean
        B, L, N, _ = y_batch.shape
        m = preds.mean.view(B, N, L, 1).permute(0, 2, 1, 3)
        pred_mean = m.squeeze(-1)      # [B, L, N]
        true_vals = y_batch.squeeze(-1)
        error, _ = mae(pred_mean, true_vals)
        
        # backward + clip + step
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # scheduler step per config
        scheduler.step()

        train_loss_sum += loss.item()
        train_error_sum += error.item()
        total_iter += 1

    avg_train_loss = train_loss_sum / len(train_loader)
    avg_train_error = train_error_sum / len(train_loader)

    return avg_train_loss, avg_train_error, total_iter


def validate_multigraphwavenet(model, val_loader, config, device, criterion=criterion, mae=mae):
    """
    Run one validation epoch, computing per-lead-time CRPS & MAE, logging to wandb.

    Returns:
        avg_val_loss, avg_val_acc, avg_crps_dict, avg_mae_dict
    """
    model.eval()
    val_loss_sum = 0.0
    val_acc_sum = 0.0
    # accumulate sums and counts for horizons
    horizons = [1, 24, 48, 96]
    sum_crps = {h: 0.0 for h in horizons}
    sum_mae = {h: 0.0 for h in horizons}
    count = {h: 0 for h in horizons}

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_in = x_batch.permute(0, 3, 2, 1)
            preds = model(x_in)

            # overall CRPS & mean-based accuracy
            loss = criterion(preds, y_batch)
            B, L, N, _ = y_batch.shape
            m = preds.mean.view(B, N, L, 1).permute(0, 2, 1, 3)
            pred_mean = m.squeeze(-1)
            true_vals = y_batch.squeeze(-1)
            acc, _ = mae(pred_mean, true_vals)

            val_loss_sum += loss.item()
            val_acc_sum += acc.item()

            # per-horizon metrics
            for h in horizons:
                idx = h
                y_h = y_batch[:, idx, :, :].unsqueeze(1)
                crps_h = criterion(preds, y_h, t=idx).item()
                # count valid points
                valid = int(torch.isfinite(y_h).sum().item())
                sum_crps[h] += crps_h * valid

                # mae at horizon
                m_h = preds.mean.view(B, N, L, 1).permute(0, 2, 1, 3)[:, idx, :, :].squeeze(-1)
                y_true_h = y_h.squeeze(-1).squeeze(1)
                err_h, cnt_h = mae(m_h, y_true_h)
                sum_mae[h] += err_h * cnt_h
                count[h] += cnt_h

    # compute averages
    avg_val_loss = val_loss_sum / len(val_loader)
    avg_val_acc = val_acc_sum / len(val_loader)
    avg_crps = {h: sum_crps[h] / count[h] for h in horizons}
    avg_mae = {h: sum_mae[h] / count[h] for h in horizons}

    return avg_val_loss, avg_val_acc, avg_crps, avg_mae