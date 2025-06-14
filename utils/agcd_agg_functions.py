import xarray as xr
import dask.array as da
import numpy as np
import dask
from rasterio.features import geometry_mask
from shapely.geometry import shape, mapping
import pandas as pd
import dill
import os

dask.config.set(array__slicing__split_large_chunks=True)

# Create a 'summer_year' coordinate that assigns December to the NEXT year
def assign_summer_year(time):
    """Assign December to the following summer year."""
    year = time.dt.year
    month = time.dt.month
    return xr.where(month == 12, year + 1, year)

def checkpoint_update(file_path, new_data, overwrite=False):
    """
    Update or overwrite a dill-based checkpoint (.pkl file) with new dictionary content.

    Parameters
    ----------
    file_path : str
        Path to the pickle file to update.
    new_data : dict
        Dictionary of new key-value pairs to save or merge into the existing file.
    overwrite : bool, optional
        If True, completely overwrite the existing file. If False (default), merge into existing content.
    """
    if overwrite or not os.path.exists(file_path):
        # Write from scratch
        with open(file_path, 'wb') as f:
            dill.dump(new_data, f)
    else:
        # Load existing and update
        with open(file_path, 'rb') as f:
            existing_data = dill.load(f)
        existing_data.update(new_data)
        # Save back safely
        tmp_path = file_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            dill.dump(existing_data, f)
        os.replace(tmp_path, file_path)

def load_var(path, varname, bbox_geom=None, chunk=True):
    """
    Load a climate variable from NetCDF files, standardising units, spatial dimensions, 
    and optionally clipping to a bounding box.

    Parameters:
    ----------
    path : str
        File path or glob pattern for NetCDF file(s).
    varname : str
        Name of the variable to extract.
    bbox_geom : list of dicts, optional
        A GeoJSON-style geometry list (e.g., [mapping(bbox_geom)]) to clip to.

    Returns:
    -------
    xarray.DataArray
        Variable loaded from file with appropriate unit conversion, dimension fixing, and spatial clipping.
    """
    if chunk:
        ds = xr.open_mfdataset(path, combine='by_coords', chunks={'time': 365})[varname]
    else:
        ds = xr.open_mfdataset(path, combine='by_coords')[varname]

    if ds.rio.crs is None:
        ds.rio.write_crs('EPSG:4326', inplace=True)

    # Optional spatial clipping
    if bbox_geom is not None:

        # Extract bounding box bounds from GeoJSON-like object
        bounds = shape(bbox_geom[0]).bounds  # (minx, miny, maxx, maxy)
        min_lon, min_lat, max_lon, max_lat = bounds

        ds = ds.sel(
            lat=slice(min_lat, max_lat),
            lon=slice(min_lon, max_lon)
        )
        #print(f"{varname} sliced shape:", ds.shape, "max:", ds.max().values)

    return ds



def calculate_avg_hot_days(x_dset, threshold=40):
    """
    Calculate the average number of hot days per year, assuming 365-day pseudo-years.

    Parameters:
    ----------
    x_dset : xarray.Dataset
        Dataset with 'tasmax' variable and daily data in multiples of 365.
    threshold : float
        Temperature threshold (°C) to count as a hot day.

    Returns:
    -------
    xarray.DataArray
        Array of average number of hot days per pseudo-year (dims: lat, lon).
    """
    tasmax = x_dset['tasmax']
    
    # Count hot days and average over years
    hot_days_per_year = (tasmax > threshold).groupby('time.year').sum(dim='time')
    return hot_days_per_year.mean(dim='year')


def calculate_percentile(x_dset, x_var='tasmax', percentile=95):
    """
    Calculate the specified temperature percentile across all years for each grid cell.

    Parameters:
    ----------
    x_dset : xarray.Dataset
        Chunked xarray dataset
    x_var : name of variable in x_dset to get percentile for
    percentile : float
        Percentile to compute (0–100), default is 95.

    Returns:
    -------
    xarray.DataArray
        2D array (lat, lon) of the percentile variable across all time.
    """
    da_var = x_dset[x_var]

     # Force single chunk along time (loads full time series per grid cell)
    #  Chunk over lat lon instead.
    da_var = da_var.chunk({'time': -1, 'lat': 50, 'lon': 50})

    # Apply along time axis
    percentile_out = xr.apply_ufunc(
        np.nanpercentile,
        da_var,
        percentile,
        input_core_dims=[["time"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da_var.dtype]
    )

    return percentile_out  # dims: (lat, lon)

def calculate_daily_mean_temp(x_dset):
    """
    Calculate the daily mean temperature at each grid cell using tasmin and tasmax NetCDFs.

    Parameters:
    ----------
    x_dset: XArray Dataset with minimum (tasmin) and maximum (tasmax) variables
    chunks : dict
        Chunking scheme for xarray/Dask (default: 365-day chunks).

    Returns:
    -------
    xarray.DataArray
        3D array (time, lat, lon) of daily mean temperatures.
    """
    # Open datasets with Dask chunking

    tasmin = x_dset['tasmin']
    tasmax = x_dset['tasmax']

    # Check dimensions and coordinates match
    if tasmin.shape != tasmax.shape:
        raise ValueError("tasmin and tasmax must have the same shape.")
    if not all(tasmin[dim].equals(tasmax[dim]) for dim in ['time', 'lat', 'lon']):
        raise ValueError("Mismatch in coordinates between tasmin and tasmax.")

    # Compute daily mean temperature
    tasmean = (tasmin + tasmax) / 2.0
    tasmean.name = 'tasmean'

    return tasmean  # dims: (time, lat, lon)

def pop_weighted_mean(data, pop):
    """
    Calculate the population-weighted average of raster values within a given zone.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of raster values (e.g., climate data) for grid cells in the zone.
    pop : numpy.ndarray
        1D array of population values corresponding to each grid cell.

    Returns
    -------
    float
        Population-weighted average of the input data. Returns np.nan if no valid
        data-population pairs are available.
    """
    # Identify grid cells with valid data and population
    valid = ~np.isnan(data) & ~np.isnan(pop)

    # If no valid overlapping data, return nan
    if valid.sum() == 0:
        return np.nan

    # Return population-weighted average across valid cells
    return np.average(data[valid], weights=pop[valid])

def zonal_weighted_mean(raster, pop_array, shapes, affine):
    """
    Compute population-weighted means for each polygonal zone over a raster.

    Parameters
    ----------
    raster : numpy.ndarray
        2D array of climate data (e.g., temperature or percentiles).
    pop_array : numpy.ndarray
        2D array of population data aligned with the raster.
    shapes : list of dicts
        List of GeoJSON-like geometries (e.g., from shapely.mapping()) representing polygons.
    affine : affine.Affine
        Affine transformation of the raster (same for both `raster` and `pop_array`).

    Returns
    -------
    list of floats
        Population-weighted average of raster values for each zone.
        Returns np.nan for zones with no valid data.
    """

    # Reproject raster to match pop if needed
    if raster.shape != pop_array.shape:
        raster = raster.rio.reproject_match(pop_array)
    
    stats = []

    # Drop extra dims (e.g., time, x/y)
    #raster = raster.squeeze()
    #pop_array = pop_array.squeeze()

    # Ensure we're working with lat/lon only
    #raster = raster.rename({k: v for k, v in zip(raster.dims, ['lat', 'lon'])})
    #pop_array = pop_array.rename({k: v for k, v in zip(pop_array.dims, ['lat', 'lon'])})

    for shape in shapes:
        # Build 2D mask
        mask = geometry_mask([shape], transform=affine, invert=True, out_shape=raster.shape)

        # Use DataArrays for clarity if desired (or work with NumPy directly)
        masked_raster = raster.where(mask)
        masked_pop = pop_array.where(mask)

        # Convert to NumPy and apply mask
        data_np = masked_raster.values
        pop_np = masked_pop.values

        # Ensure valid data only
        valid = ~np.isnan(data_np) & (data_np > 0) & ~np.isnan(pop_np) & (pop_np > 0)

        if np.any(valid):
            stats.append(np.average(data_np[valid], weights=pop_np[valid]))
        else:
            stats.append(np.nan)


    return stats

def calculate_heat_index(tasmax_c, rh, threshold_temp=27, threshold_rh=40):
    """
    Calculate NOAA Heat Index (in °C) from daily maximum temperature and relative humidity.

    Parameters:
        tasmax_c (xarray.DataArray): Daily maximum air temperature in degrees Celsius.
        rh_percent (xarray.DataArray): Daily relative humidity in percent (0–100%).

    Returns:
        xarray.DataArray: Apparent temperature (Heat Index) in degrees Celsius.
                          Values are only calculated where temperature >= 27°C and RH >= 40%;
                          other values are returned as NaN.

    Notes:
        - The Heat Index is calculated using the official NOAA multiple regression formula in °F.
        - Temperature is converted from °C to °F for the calculation, then back to °C.
        - Adjustments are applied for low RH (T: 80–112°F, RH < 13%) and high RH (T: 80–87°F, RH > 85%) only.
        - Outputs are masked to avoid over-applying the formula outside its valid domain.
    """
    # Convert to Fahrenheit
    T = tasmax_c * 9 / 5 + 32
    RH = rh

    # Define valid range mask for main HI calculation
    valid = (T >= 80) & (RH >= 40)

    # Initialise HI with NaNs
    HI = xr.full_like(T, np.nan)

    # Base NOAA HI formula (F) — apply only where valid
    T_valid = T.where(valid)
    RH_valid = RH.where(valid)

    base_HI = (
        -42.379 +
        2.04901523 * T_valid +
        10.14333127 * RH_valid -
        0.22475541 * T_valid * RH_valid -
        0.00683783 * T_valid**2 -
        0.05481717 * RH_valid**2 +
        0.00122874 * T_valid**2 * RH_valid +
        0.00085282 * T_valid * RH_valid**2 -
        0.00000199 * T_valid**2 * RH_valid**2
    )

    HI = xr.where(valid, base_HI, HI)

    # LOW RH adjustment
    low_mask = (RH < 13) & (T >= 80) & (T <= 112)
    low_adj = xr.where(low_mask, ((13 - RH) / 4) * np.sqrt((17 - abs(T - 95)) / 17), 0.0)
    HI -= low_adj

    # HIGH RH adjustment
    high_mask = (RH > 85) & (T >= 80) & (T <= 87)
    high_adj = xr.where(high_mask, ((RH - 85) / 10) * ((87 - T) / 5), 0.0)
    HI += high_adj
    # Final conversion to °C
    HI_c = (HI - 32) * 5 / 9

    return HI_c

def calculate_ehf(tmean, t95_hist, rolling_3day=3, rolling_30day=30):
    """
    Calculate Excess Heat Factor (EHF) using xarray, based on fixed historical T95.

    Parameters
    ----------
    tmean : xarray.DataArray
        Daily mean temperature (tasmax + tasmin) / 2, for future or observed period.
    t95_hist : xarray.DataArray
        Fixed historical 95th percentile daily mean temperature (lat, lon).
    rolling_3day : int
        Window for 3-day mean (default = 3).
    rolling_30day : int
        Window for 30-day acclimatisation mean (default = 30).

    Returns
    -------
    xarray.DataArray
        EHF values (same dims as tmean), NaN outside valid rolling windows.
    """
    #t95_hist = t95_hist.rio.reproject_match(tmean)

    #t95_hist = t95_hist.rename({'x': 'lon', 'y': 'lat'})

    # Rolling windows
    t3 = tmean.rolling(time=rolling_3day, center=False).mean()
    t30 = tmean.rolling(time=rolling_30day, center=False, min_periods=int(0.9 * rolling_30day)).mean()

    ehisig = t3 - t95_hist
    ehiaccl = t3 - t30
    ehf = ehisig * xr.where(ehiaccl > 0, ehiaccl, 1)

    return ehf

def calculate_avg_ehf_days(tmean, t95_hist, rolling_3day=3, rolling_30day=30):
    """
    Calculate the average number of EHF days per year.

    Parameters:
    ----------
    tmean : xarray.DataArray
        Daily mean temperature.
    t95_hist : xarray.DataArray
        Historical 95th percentile threshold.
    rolling_3day : int
        Window for 3-day mean.
    rolling_30day : int
        Window for 30-day mean.

    Returns:
    -------
    xarray.DataArray
        Average number of EHF days per year (lat, lon).
    """
    
    ehf = calculate_ehf(tmean, t95_hist, rolling_3day, rolling_30day)

    try:
        years = [t.year for t in ehf['time'].values]
    except AttributeError:
        years = [pd.Timestamp(t).year for t in ehf['time'].values]
    
    ehf = ehf.assign_coords(year=('time', years))

    ehf_days_per_year = (ehf > 0).groupby('year').sum(dim='time')
    avg_ehf_days = ehf_days_per_year.mean(dim='year')

    return avg_ehf_days

def summarise_heatwaves(ehf, threshold=0):
    """
    Summarise heatwave statistics (average events/year and duration) from EHF data.

    Parameters
    ----------
    ehf : xarray.DataArray
        Daily EHF time series with 'time' coordinate.
    threshold : float, optional
        EHF threshold to define a heatwave day (default = 0).

    Returns:
    -------
    dict of xarray.DataArray
        {
            'avg_events_per_year': xarray.DataArray (lat/lon),
            'avg_duration': xarray.DataArray (lat/lon)
        }
    """

    # Ensure year is a coordinate
    if 'year' not in ehf.coords:
        years = xr.DataArray(ehf['time'].dt.year.values, dims='time', name='year')
        ehf = ehf.assign_coords(year=years)

    # Group by year
    grouped = ehf.groupby('year')

    def per_year_stats(ehf_1d, threshold=0):
        """
        Count heatwaves and duration in a 1D EHF series for a single year.
        """
        arr = np.asarray(ehf_1d > threshold, dtype=bool)

        diffs = np.diff(np.concatenate([[0], arr.view(np.int8), [0]]))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        lengths = ends - starts

        valid_lengths = lengths[lengths >= 3]
        count = len(valid_lengths)
        avg_dur = valid_lengths.mean() if count > 0 else 0.0
        return count, avg_dur

    # Apply per year
    counts, durations = xr.apply_ufunc(
        per_year_stats,
        grouped,
        kwargs={'threshold': threshold},
        input_core_dims=[['time']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int, float],
    )

    # Take average across years
    avg_events_per_year = counts.mean(dim='year')
    avg_duration = durations.where(counts > 0).mean(dim='year')  # avoid dividing by 0

    return {
        'avg_events_per_year': avg_events_per_year,
        'avg_duration': avg_duration
    }

def calculate_mean(var):
    """
    Compute the yearly mean of a time-series xarray.DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray with a 'time' coordinate (datetime-like).

    Returns
    -------
    xarray.DataArray
        Yearly mean values with 'year' as the new time dimension.
    """

    # assign year
    try:
        years = [t.year for t in var['time'].values]
    except AttributeError:
        years = [pd.Timestamp(t).year for t in var['time'].values]

    var = var.assign_coords(year=('time', years))

    return var.groupby('year').mean(dim='time')


def weighted_median(data, weights):
    """
    Compute the weighted median of data given associated weights.
    """
    # Flatten and filter
    data, weights = np.array(data).flatten(), np.array(weights).flatten()
    valid = (~np.isnan(data)) & (~np.isnan(weights)) & (weights > 0)

    if not np.any(valid):
        return np.nan

    data, weights = data[valid], weights[valid]

    # Sort by data values
    sorted_indices = np.argsort(data)
    data_sorted = data[sorted_indices]
    weights_sorted = weights[sorted_indices]

    # Compute cumulative weight
    cum_weights = np.cumsum(weights_sorted)
    cutoff = 0.5 * np.sum(weights_sorted)

    # Find index where cumulative weight exceeds half the total weight
    median_index = np.searchsorted(cum_weights, cutoff)
    return data_sorted[median_index]

def export_raster(ds, output_path, x='lon', y='lat', crs='EPSG:4326'):
    """
    Safely write a DataArray to a GeoTIFF, ensuring spatial dims and CRS are set.

    Parameters
    ----------
    ds : xr.DataArray
        The DataArray to write.
    output_path : str
        The file path to write the GeoTIFF to.
    x, y : str
        Names of the x and y spatial dimensions (default 'lon', 'lat').
    crs : str
        CRS to assign if not already present (e.g., 'EPSG:4326').
    """
    if not ds.rio.crs:
        ds = ds.rio.write_crs(crs, inplace=False)

    try:
        _ = ds.rio.x_dim
        _ = ds.rio.y_dim
    except:
        ds.rio.set_spatial_dims(x_dim=x, y_dim=y, inplace=True)

    ds.rio.to_raster(output_path)


def zonal_weighted_mean_time_series(ncdf, weights, zones, affine, idcol):
    """
    Compute daily population-weighted means for each polygonal zone using 
    Dask-aware xarray operations.

    Parameters
    ----------
    ncdf: xarray.DataArray
        Climate variable NetCDF with dimensions (time, lat, lon), e.g., daily sfcWindmax.
    weights : xarray.DataArray
        2D array of aligned weights (lat, lon).
    zones : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries representing each zone
        (e.g., health districts).
    affine : affine.Affine
        Affine transformation for the spatial grid (must match raster and pop_array).
    idcol : string
        zone ID column name to assign to output dataframe column names

    Returns
    -------
    pandas.DataFrame
        A DataFrame with dates as the index and one column per health district,
        containing the population-weighted mean value for each day and zone.
    """
    shapes = [mapping(geom) for geom in zones.geometry]

    zone_results = {}

    for i, zone in enumerate(shapes):
        mask = geometry_mask([zone], transform=affine, invert=True, out_shape=weights.shape)
        mask_da = xr.DataArray(mask, coords=weights.coords, dims=weights.dims)

        # Slice the original arrays to avoid broadcasting huge arrays
        masked_data = ncdf.where(mask_da)
        masked_weights = weights.where(mask_da)

        # Now chunk per-zone after masking
        masked_data = masked_data.chunk({'lat': 'auto', 'lon': 'auto'})
        masked_weights = masked_weights.chunk({'lat': 'auto', 'lon': 'auto'})

        weighted_mean = (masked_data * masked_weights).sum(dim=['lat', 'lon']) / masked_weights.sum(dim=['lat', 'lon']).compute()

        zone_name = zones.iloc[i][idcol]
        #print(f'{zone_name} computed')
        
        zone_results[zone_name] = weighted_mean.to_series()

    df = pd.DataFrame(zone_results)
    df.index.name = 'date'
    
    return df.reset_index()

def zonal_mean_time_series(ncdf, zones, affine, idcol):
    """
    Compute daily unweighted mean for each polygonal zone

    Parameters
    ----------
    ncdf : xarray.DataArray
        Climate variable with dimensions (time, lat, lon).
    zones : geopandas.GeoDataFrame
        Zones for aggregation.
    affine : affine.Affine
        Affine transform matching ncdf grid.
    idcol : str
        Name of the column with zone identifiers.

    Returns
    -------
    pandas.DataFrame
        Time series DataFrame with one column per zone.
    """
    shapes = [mapping(geom) for geom in zones.geometry]
    results = {}

    for i, geom in enumerate(shapes):
        zone_name = zones.iloc[i][idcol]

        mask = geometry_mask([geom], transform=affine, invert=True, out_shape=ncdf.shape[1:])
        mask_da = xr.DataArray(mask, coords={'lat': ncdf.lat, 'lon': ncdf.lon}, dims=('lat', 'lon'))

        mean_ts = ncdf.where(mask_da).mean(dim=['lat', 'lon'])

        results[zone_name] = mean_ts.compute().to_series()

    df = pd.DataFrame(results)
    df.index.name = 'date'


def zonal_maximum_time_series(ncdf, zones, affine, idcol):
    """
    Compute daily maximum for each polygonal zone

    Parameters
    ----------
    ncdf : xarray.DataArray
        Climate variable with dimensions (time, lat, lon).
    zones : geopandas.GeoDataFrame
        Zones for aggregation.
    affine : affine.Affine
        Affine transform matching ncdf grid.
    idcol : str
        Name of the column with zone identifiers.

    Returns
    -------
    pandas.DataFrame
        Time series DataFrame with one column per zone.
    """
    shapes = [mapping(geom) for geom in zones.geometry]
    results = {}

    for i, geom in enumerate(shapes):
        zone_name = zones.iloc[i][idcol]

        mask = geometry_mask([geom], transform=affine, invert=True, out_shape=ncdf.shape[1:])
        mask_da = xr.DataArray(mask, coords={'lat': ncdf.lat, 'lon': ncdf.lon}, dims=('lat', 'lon'))

        max_ts = ncdf.where(mask_da).max(dim=['lat', 'lon'])

        results[zone_name] = max_ts.compute().to_series()

    df = pd.DataFrame(results)
    df.index.name = 'date'
    return df.reset_index()







