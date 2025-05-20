
def train_and_save(dm, edge_index, config, epochs=100, anomalous_mask=True, hyperparam_search=False, verbose=True, device=torch.device('cpu')):
    """
    dm            : PostprocessDatamodule
    edge_index    : from adj_to_edge_index(dm.adj_matrix)
    config        : dict with keys:
      - "model": one of {"baseline","enhanced_tcn_gnn",}
      - hyperparams: learning rate, weight_decay, hidden_size, num_layers, dropout_p, scheduler type & params
    epochs        : how many epochs to train
    """

    model_name    = config["model"]
    lr            = config["lr"]
    wd            = config["weight_decay"]
    hs            = config["hidden_size"]
    nl            = config["num_layers"]
    dp            = config["dropout_p"]
    sch_name = config["scheduler"]
    if model_name != "baseline":
        kernel_size = config.get("kernel_size",3)

    # instantiate model
    if model_name == "baseline":
        model = EnhancedGRUBaseline(
            input_size = dm.train_dataset.features,
            hidden_channels = hs,
            output_dist = "LogNormal",
            n_stations = dm.train_dataset.stations,
            num_layers = nl,
            dropout_p = dp
        )
    elif model_name == "enhanced_tcn_gnn":
        model = EnhancedTCNGNN(input_size=dm.train_dataset.features,
                               hidden_channels=hs,
                               output_dist="LogNormal",
                               n_stations=dm.train_dataset.stations,
                               num_layers=nl,
                               kernel_size=kernel_size,
                               dropout_p=dp,)

    else:
        raise ValueError(f"Unknown model {model_name}")

    model.to(device)
    model.apply(init_weights)
    edge_index = edge_index.to(device)

    # data loaders
    train_dl = DataLoader(dm.train_dataset, batch_size=config["batch_size"], shuffle=True,)
    val_dl   = DataLoader(dm.val_dataset,   batch_size=config["batch_size"], shuffle=False,)

    # optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if sch_name == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_dl),
            pct_start=config.get("pct_start",0.3)
        )
    elif sch_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config["T_max"]
        )
    elif sch_name == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config["gamma"]
        )
    elif sch_name == "exponential":
        scheduler = ExponentialLR(
            optimizer,
            gamma=config["exp_gamma"]
        )
    elif sch_name == "cosinewarm":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["T_0"],
            T_mult=config.get("T_mult",1),
            eta_min=config.get("eta_min",1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler {sch_name}")

    # losses
    crps_crit = MaskedCRPSLogNormal()
    mae_crit  = MaskedL1Loss()

    horizons = [1,24,48,96]
    best_val_crps = float("inf")
    best_val_crps_h = {h: float("inf") for h in horizons}
    best_val_mae_h  = {h: float("inf") for h in horizons}
    best_state_dict = None
    early_stopping_counter = 0

    # output folders
    os.makedirs(MODELS_PATH, exist_ok=True)    #TODO: Change Path
    os.makedirs(HYPERPARAMS_PATH, exist_ok=True)   #TODO: Change Path
    os.makedirs(f"{PRED_PLOT_PATH}/{model_name}", exist_ok=True)    #TODO: Change Path
    os.makedirs(f"{LOSS_PATH}/{model_name}", exist_ok=True)   #TODO: Change Path

    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_crps_sum = 0.0
        train_mae_sum  = 0.0
        torch.cuda.empty_cache()
        for i, (x,y) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch} [Train]")):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # forward
            dist = model(x, edge_index) if model_name!="baseline" else model(x)
            if anomalous_mask:
                y = mask_anomalous_targets(y, min_speed=0.2, max_speed=10.0)

            loss = crps_crit(dist, y)
            mu   = dist.mean.squeeze(-1)
            mae  = mae_crit(mu, y.squeeze(-1))
            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            # scheduler step
            if sch_name=="cosinewarm":
                scheduler.step(epoch + i/len(train_dl))
            else:
                scheduler.step()

            train_crps_sum += loss.item()
            train_mae_sum  += mae.item()

        avg_train_crps = train_crps_sum / len(train_dl)
        avg_train_mae  = train_mae_sum  / len(train_dl)

        # VALIDATION
        model.eval()
        val_crps_sum = 0.0
        val_mae_sum  = 0.0
        sum_crps     = {h:0.0 for h in horizons}
        cnt_crps     = {h:0   for h in horizons}
        sum_mae      = {h:0.0 for h in horizons}
        cnt_mae      = {h:0   for h in horizons}

        with torch.no_grad():
            for x,y in tqdm(val_dl, desc=f"Epoch {epoch} [Valid]"):
                x,y = x.to(device), y.to(device)
                dist = model(x, edge_index) if model_name!="baseline" else model(x)
                if anomalous_mask:
                    y = mask_anomalous_targets(y, min_speed=0.2, max_speed=10.0)
                loss = crps_crit(dist, y)
                mu   = dist.mean.squeeze(-1)
                mae  = mae_crit(mu, y.squeeze(-1))
                val_crps_sum += loss.item()
                val_mae_sum  += mae.item()

                # per-horizon
                for h in horizons:
                    loc_h   = dist.loc[:,h,:,:].squeeze(-1)
                    sc_h    = dist.scale[:,h,:,:].squeeze(-1)
                    d_h     = torch.distributions.LogNormal(loc_h,sc_h)
                    y_h     = y[:,h,:,:].squeeze(-1)
                    valid   = (~torch.isnan(y_h)).sum().item()
                    c_h     = crps_crit(d_h, y_h).item()
                    m_h     = mae_crit(mu[:,h,:], y_h).item()
                    sum_crps[h] += c_h*valid
                    cnt_crps[h] += valid
                    sum_mae [h] += m_h*valid
                    cnt_mae [h] += valid

        avg_val_crps = val_crps_sum / len(val_dl)
        avg_val_mae  = val_mae_sum  / len(val_dl)
        avg_crps_h   = {h: sum_crps[h]/cnt_crps[h] for h in horizons}
        avg_mae_h    = {h: sum_mae [h]/cnt_mae [h] for h in horizons}

        # ——— LOG & PRINT —————————————————————————————————————
        if verbose and not hyperparam_search:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f" Train →  CRPS: {avg_train_crps:.4f} │ MAE: {avg_train_mae:.4f}")
            print(f" Valid →  CRPS: {avg_val_crps:.4f} │ MAE: {avg_val_mae:.4f}\n")
            print(" Horizon │   CRPS   │    MAE ")
            print("-"*32)
            for h in horizons:
                print(f" {h:6d} │ {avg_crps_h[h]:8.4f} │ {avg_mae_h[h]:8.4f}")
            print()

        # ——— log_prediction_plots on last val batch ——————————————————
        if epoch == epochs - 1 and not hyperparam_search:
            log_prediction_plots(
                x = x.cpu(),
                y = y.cpu(),
                pred_dist = dist,
                example_indices = [0,0,0,0],
                stations        = [0,1,2,3],
                epoch           = epoch,
                input_denormalizer = dm.val_dataset.input_denormalizer,
                folder_name = PRED_PLOT_PATH,     #TODO: Change Path or make it a parameter
                model_name = model_name,
                filename=f"epoch_{epoch+1}"
            )

        # ——— save best model + config —————————————————————————————————
        if avg_val_crps < best_val_crps:
            best_val_crps = avg_val_crps
            best_val_crps_h = avg_crps_h
            best_val_mae_h  = avg_mae_h
            best_state_dict = model.state_dict()
            early_stopping_counter = 0

        if early_stopping_counter == 10:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        else:
            early_stopping_counter += 1

    if verbose and not hyperparam_search:
        print("Best validation CRPS:", best_val_crps)

    checkpoint_path = os.path.join(MODELS_PATH, f'{model_name}_best.pth')
    torch.save(best_state_dict, checkpoint_path)
    #torch.save({
    #    'epoch': epoch,
    #    'model_state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #}, checkpoint_path)
    
    return best_state_dict, best_val_crps, best_val_crps_h, best_val_mae_h