def mask_anomalous_targets(y, min_speed=0.1, max_speed=60.0):
    """
    y: tensor of shape [B, L, N, 1] (or [B, L, N]) containing observed wind speeds
    min_speed: speeds below this (e.g. frozen sensors) will be masked
    max_speed: speeds above this (physically implausible gusts) will be masked
    returns: a copy of y with out‐of‐range values replaced by NaN
    """
    squeezed = (y.squeeze(-1) if y.dim()==4 else y)
    bad = (squeezed < min_speed) | (squeezed > max_speed) | torch.isnan(squeezed)
    y_clean = squeezed.clone()
    y_clean[bad] = float('nan')
    return y_clean.unsqueeze(-1) if y.dim()==4 else y_clean



def init_weights(m):
    """
    Initialize module weights to promote stable and efficient training across diverse layer types.

    The Xavier (Glorot) uniform initialization for embeddings and fully connected layers preserves variance
    of activations and gradients when propagating through depth, preventing early saturation or vanishing/exploding
    gradients. Zeroing biases ensures no unwanted initial offsets. For convolutional layers, Kaiming (He) uniform
    initialization tailored to ReLU non-linearities maintains the forward signal’s scale and supports deeper
    temporal convolutional networks.

    Recurrent layers (GRUs) benefit from a hybrid scheme: input-to-hidden weights follow Xavier uniform to
    balance inputs, while hidden-to-hidden weights use orthogonal initialization. Orthogonal matrices preserve
    vector norms under repeated multiplication, which mitigates gradient decay or explosion over long sequences,
    improving temporal memory. All GRU biases start at zero to avoid biasing update gates at initialization.

    BatchNorm weights are set to one (identity scaling) and biases to zero so that it doesn't normalize it in the beginning,
    letting subsequent layers learn necessary adjustments. Multihead attention modules require careful splitting:
    the combined in_proj matrix is Xavier initialized to treat query, key, and value uniformly, and the output
    projection is also Xavier initialized, with all biases zeroed. A catch all clause ensures any custom layer
    with weight/bias attributes receives a sensible Xavier or zero initialization, avoiding uninitialized parameters.
    """
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        if m.out_proj.weight is not None:
            nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)

    elif hasattr(m, 'weight') and m.weight is not None:
        if m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.zeros_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)



