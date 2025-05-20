import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tsl.ops.similarities import top_k
import xarray as xr
from typing import Union, Optional
from omegaconf import ListConfig
import numpy as np

class XarrayDataset(Dataset):
    def __init__(self, input_data, target_data):
        
        self.input_data, self.input_denormalizer = self.normalize(np.transpose(input_data.to_array().data, (1,2,3,0)))
        
        # NOTE No transformation is applied to targets. If normalized ~(0,1), they can be negatve
        # which is incompatible with CRPS_LogNormal. 
        self.target_data, self.target_denormalizer = np.transpose(target_data.to_array().data, (1,2,3,0)), lambda x : x
        
        
        self.t, self.l, self.s, self.f = self.input_data.shape
        self.tg = self.target_data.shape[-1]
        
    def normalize(self, data):
        data_mean = np.nanmean(data, axis=(0, 1, 2), keepdims=True)
        data_std = np.nanstd(data, axis=(0, 1, 2), keepdims=True)
        standardized_data = (data - data_mean) / data_std
        
        def denormalizer(x): # closure (alternative: use a partial)
            if isinstance(x, torch.Tensor):
                return (x * torch.Tensor(data_std).to(x.device)) + torch.Tensor(data_mean).to(x.device)
            return (x * data_std) + data_mean
        
        return standardized_data, denormalizer
    
    def get_baseline_score(self, score_fn):
        pass
    
    @property
    def stations(self):
        return self.s 
    
    @property
    def forecasting_times(self):
        return self.t
    
    @property
    def lead_times(self):
        return self.l
    
    @property
    def features(self):
        return self.f
    
    @property
    def targets(self):
        return self.tg
        
    def __len__(self):
        return self.input_data.shape[0]  # Number of forecast_reference_time
    
    def __getitem__(self, idx):
        sample_x = self.input_data[idx]
        sample_y = self.target_data[idx]
        return torch.tensor(sample_x, dtype=torch.float), torch.tensor(sample_y, dtype=torch.float)
    
    
    def __str__(self):
        return f"Dataset: [time={self.t}, lead_time={self.l}, stations={self.s}, features={self.f}] | target dim={self.tg}\n" 
    
    
    
def get_graph(lat, lon, knn=10, threshold=None, theta=None):
    
    def haversine(lat1, lon1, lat2, lon2, radius=6371):
        import math
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Differences in coordinates
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = radius * c

        return distance
    n = lat.shape[0]
    dist = np.zeros((n,n))
    for i in tqdm(range(n), desc="Outer Loop Progress"):
        for j in tqdm(range(i, n), desc=f"Inner Loop Progress (i={i})", leave=False):
            s1_lon = lon[i]
            s1_lat = lat[i]
            
            s2_lon = lon[j]
            s2_lat = lat[j]
            
            d = haversine(lat1=s1_lat, lon1=s1_lon, lat2=s2_lat, lon2=s2_lon)
            dist[i,j] = d
            dist[j,i] = d
            
    def gaussian_kernel(x, theta=None):
        if theta is None or theta == "std":
            theta = x.std()
        elif theta == "median":
            # extract strictly off-diagonal entries
            i, j = np.triu_indices(dist.shape[0], k=1)
            d_off = dist[i, j]
            theta = np.median(d_off)
        elif theta == "factormedian":
            # extract strictly off-diagonal entries
            i, j = np.triu_indices(dist.shape[0], k=1)
            d_off = dist[i, j]
            theta = np.median(d_off)*0.5
        print(type(theta), type(x))
        weights = np.exp(-np.square(x / theta))
        return weights
    
    adj = gaussian_kernel(dist, theta)
    
    adj = top_k(adj,knn, include_self=True,keep_values=True)
    
    if threshold is not None:
            adj[adj < threshold] = 0
    
    return adj

class PostprocessDatamodule():
    def __init__(self, train_dataset: XarrayDataset, 
                 val_dataset: XarrayDataset, 
                 test_dataset: XarrayDataset, 
                 adj_matrix: np.ndarray = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.adj_matrix = adj_matrix
        self.num_edges = (self.adj_matrix != 0).astype(np.int32).sum() if adj_matrix is not None else 0

    
    def __str__(self) -> str:
        dm_str = "Data Module: \n\n"
        dm_str += "Train:\n"
        dm_str += str(self.train_dataset)
        dm_str += "Val:\n"
        dm_str += str(self.val_dataset)
        dm_str += "Test:\n"
        dm_str += str(self.test_dataset)
        
        dm_str += f"Number of edges = {self.num_edges}"
        return dm_str

def get_datamodule(ds: xr.Dataset, 
                   ds_targets: xr.Dataset,
                   predictors: Union[list, ListConfig],
                   lead_time_hours: int,
                   val_split: float,
                   target_var: str, 
                   test_start_date: str,
                   train_val_end_date: Optional[str] = None,
                   return_graph=True, 
                   graph_kwargs=None) -> PostprocessDatamodule:
    """_summary_

    Args:
        ds (xr.Dataset): The input dataset.
        ds_targets (xr.Dataset): The target dataset.
        predictors (Union[list, ListConfig]): The variable names to be used as predictors.
        lead_time_hours (int): The number of hours considered for the forecasted window.
        val_split (float): The percentage in [0,1) to be used as validation.
        target_var (str): The (single) target variable.
        test_start_date (str): The day where the test set will start.
        train_val_end_date (Optional[str], optional): The day when train and validation end. If not provided it will be set to test_start_date - 1. 
        Pick a date that ensures no data leakage, eg. by using a large enough gap. Defaults to None.
        graph_kwargs (_type_, optional): Arguments to be passed to the graph-generating function. Defaults to None.

    Returns:
        PostprocessDatamodule: A datamodule with the train/val/test splits.
    """
    if isinstance(predictors, ListConfig):
        predictors = list(predictors)
    test_datetime = np.datetime64(test_start_date)
    if train_val_end_date is None:
        train_val_datetime = test_datetime - np.timedelta64(1,'D')
    else: 
        train_val_datetime = np.datetime64(train_val_end_date)
        
    print(f'Train&Val sets end at {train_val_datetime}')
    print(f'Test set starts at {test_datetime}')
    
    # Get input data and split 
    input_data = ds[predictors] 
    input_data = input_data.sel(lead_time=slice(None, np.timedelta64(lead_time_hours, 'h')))
    
    input_data_train_val = input_data.sel(forecast_reference_time=slice(None, train_val_datetime))
    test_input_data = input_data.sel(forecast_reference_time=slice(test_datetime, None))
    
    train_val_rtimes = len(input_data_train_val['forecast_reference_time'])
    split_index = int(train_val_rtimes * (1.0 - val_split))
    
    train_input_data = input_data_train_val.isel(forecast_reference_time=slice(0, split_index))
    val_input_data = input_data_train_val.isel(forecast_reference_time=slice(split_index, None))
    
    # Get target data
    target_data = ds_targets[[target_var]]
    target_data = target_data.sel(lead_time=slice(None, np.timedelta64(lead_time_hours, 'h')))
    
    target_data_train_val = target_data.sel(forecast_reference_time=slice(None, train_val_datetime))
    test_target_data = target_data.sel(forecast_reference_time=slice(test_datetime, None))
    
    train_target_data = target_data_train_val.isel(forecast_reference_time=slice(0, split_index))
    val_target_data = target_data_train_val.isel(forecast_reference_time=slice(split_index, None))
    
    if return_graph:
        lat = ds.latitude.data
        lon = ds.longitude.data
        adj_matrix = get_graph(lat=lat, lon=lon, **graph_kwargs)
        return PostprocessDatamodule(train_dataset=XarrayDataset(input_data=train_input_data, target_data=train_target_data),
                                     val_dataset=XarrayDataset(input_data=val_input_data, target_data=val_target_data),
                                     test_dataset=XarrayDataset(input_data=test_input_data, target_data=test_target_data),
                                     adj_matrix=adj_matrix)
    return PostprocessDatamodule(train_dataset=XarrayDataset(input_data=train_input_data, target_data=train_target_data),
                                     val_dataset=XarrayDataset(input_data=val_input_data, target_data=val_target_data),
                                     test_dataset=XarrayDataset(input_data=test_input_data, target_data=test_target_data))                        


class XarrayDatasetAnomalous(Dataset):
    def __init__(self, input_data, target_data, anomalous=False):

        self.input_data, self.input_denormalizer = self.normalize(np.transpose(input_data.to_array().data, (1,2,3,0)))

        # NOTE No transformation is applied to targets. If normalized ~(0,1), they can be negatve
        # which is incompatible with CRPS_LogNormal.
        if not anomalous:
            self.target_data, self.target_denormalizer = np.transpose(target_data.to_array().data, (1,2,3,0)), lambda x : x
        else:
        # load raw targets
        # anomalous data
            raw_y = np.transpose(target_data.to_array().data, (1,2,3,0))
            # mask out-of-range values
            clean_y = self.mask_anomalous_targets(
                torch.from_numpy(raw_y).float(),
                min_speed=0.2,
                max_speed=10.0
            ).numpy()  # still shape (t, l, s, 1) or (t,l,s)
    
            self.target_data = clean_y
            self.target_denormalizer = lambda x: x
            print("masked anomalous data")


        self.t, self.l, self.s, self.f = self.input_data.shape
        self.tg = self.target_data.shape[-1]

    def normalize(self, data):
        data_mean = np.nanmean(data, axis=(0, 1, 2), keepdims=True)
        data_std = np.nanstd(data, axis=(0, 1, 2), keepdims=True)
        standardized_data = (data - data_mean) / data_std

        def denormalizer(x): # closure (alternative: use a partial)
            if isinstance(x, torch.Tensor):
                return (x * torch.Tensor(data_std).to(x.device)) + torch.Tensor(data_mean).to(x.device)
            return (x * data_std) + data_mean

        return standardized_data, denormalizer

    def mask_anomalous_targets(self, y, min_speed, max_speed):
        squeezed = (y.squeeze(-1) if y.dim()==4 else y)
        bad = (squeezed < min_speed) | (squeezed > max_speed) | torch.isnan(squeezed)
        y_clean = squeezed.clone()
        y_clean[bad] = float('nan')
        return y_clean.unsqueeze(-1) if y.dim()==4 else y_clean

    def get_baseline_score(self, score_fn):
        pass

    @property
    def stations(self):
        return self.s

    @property
    def forecasting_times(self):
        return self.t

    @property
    def lead_times(self):
        return self.l

    @property
    def features(self):
        return self.f

    @property
    def targets(self):
        return self.tg

    def __len__(self):
        return self.input_data.shape[0]  # Number of forecast_reference_time

    def __getitem__(self, idx):
        sample_x = self.input_data[idx]
        sample_y = self.target_data[idx]
        return torch.tensor(sample_x, dtype=torch.float), torch.tensor(sample_y, dtype=torch.float)


    def __str__(self):
        return f"Dataset: [time={self.t}, lead_time={self.l}, stations={self.s}, features={self.f}] | target dim={self.tg}\n"



class PostprocessDatamoduleAnomalous():
    def __init__(self, train_dataset: XarrayDataset,
                 val_dataset: XarrayDataset,
                 test_dataset: XarrayDataset,
                 adj_matrix: np.ndarray = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.adj_matrix = adj_matrix
        self.num_edges = (self.adj_matrix != 0).astype(np.int32).sum() if adj_matrix is not None else 0


    def __str__(self) -> str:
        dm_str = "Data Module: \n\n"
        dm_str += "Train:\n"
        dm_str += str(self.train_dataset)
        dm_str += "Val:\n"
        dm_str += str(self.val_dataset)
        dm_str += "Test:\n"
        dm_str += str(self.test_dataset)

        dm_str += f"Number of edges = {self.num_edges}"
        return dm_str


def get_datamodule_anomalous(ds: xr.Dataset,
                   ds_targets: xr.Dataset,
                   predictors: Union[list, ListConfig],
                   lead_time_hours: int,
                   val_split: float,
                   target_var: str,
                   test_start_date: str,
                   train_val_end_date: Optional[str] = None,
                   return_graph=True,
                   graph_kwargs=None,
                   anomalous = False) -> PostprocessDatamodule:
    """_summary_

    Args:
        ds (xr.Dataset): The input dataset.
        ds_targets (xr.Dataset): The target dataset.
        predictors (Union[list, ListConfig]): The variable names to be used as predictors.
        lead_time_hours (int): The number of hours considered for the forecasted window.
        val_split (float): The percentage in [0,1) to be used as validation.
        target_var (str): The (single) target variable.
        test_start_date (str): The day where the test set will start.
        train_val_end_date (Optional[str], optional): The day when train and validation end. If not provided it will be set to test_start_date - 1.
        Pick a date that ensures no data leakage, eg. by using a large enough gap. Defaults to None.
        graph_kwargs (_type_, optional): Arguments to be passed to the graph-generating function. Defaults to None.

    Returns:
        PostprocessDatamodule: A datamodule with the train/val/test splits.
    """
    if isinstance(predictors, ListConfig):
        predictors = list(predictors)
    test_datetime = np.datetime64(test_start_date)
    if train_val_end_date is None:
        train_val_datetime = test_datetime - np.timedelta64(1,'D')
    else:
        train_val_datetime = np.datetime64(train_val_end_date)

    print(f'Train&Val sets end at {train_val_datetime}')
    print(f'Test set starts at {test_datetime}')

    # Get input data and split
    input_data = ds[predictors]
    input_data = input_data.sel(lead_time=slice(None, np.timedelta64(lead_time_hours, 'h')))

    input_data_train_val = input_data.sel(forecast_reference_time=slice(None, train_val_datetime))
    test_input_data = input_data.sel(forecast_reference_time=slice(test_datetime, None))

    train_val_rtimes = len(input_data_train_val['forecast_reference_time'])
    split_index = int(train_val_rtimes * (1.0 - val_split))

    train_input_data = input_data_train_val.isel(forecast_reference_time=slice(0, split_index))
    val_input_data = input_data_train_val.isel(forecast_reference_time=slice(split_index, None))

    # Get target data
    target_data = ds_targets[[target_var]]
    target_data = target_data.sel(lead_time=slice(None, np.timedelta64(lead_time_hours, 'h')))

    target_data_train_val = target_data.sel(forecast_reference_time=slice(None, train_val_datetime))
    test_target_data = target_data.sel(forecast_reference_time=slice(test_datetime, None))

    train_target_data = target_data_train_val.isel(forecast_reference_time=slice(0, split_index))
    val_target_data = target_data_train_val.isel(forecast_reference_time=slice(split_index, None))

    if return_graph:
        lat = ds.latitude.data
        lon = ds.longitude.data
        adj_matrix = get_graph(lat=lat, lon=lon, **graph_kwargs)
        return PostprocessDatamoduleAnomalous(train_dataset=XarrayDatasetAnomalous(input_data=train_input_data, target_data=train_target_data, anomalous=anomalous),
                                     val_dataset=XarrayDatasetAnomalous(input_data=val_input_data, target_data=val_target_data,anomalous=anomalous),
                                     test_dataset=XarrayDatasetAnomalous(input_data=test_input_data, target_data=test_target_data, anomalous=anomalous),
                                     adj_matrix=adj_matrix)
    return PostprocessDatamoduleAnomalous(train_dataset=XarrayDatasetAnomalous(input_data=train_input_data, target_data=train_target_data,anomalous=anomalous),
                                     val_dataset=XarrayDatasetAnomalous(input_data=val_input_data, target_data=val_target_data,anomalous=anomalous),
                                     test_dataset=XarrayDXarrayDatasetAnomalousataset(input_data=test_input_data, target_data=test_target_data,anomalous=anomalous))