"""
Combine historical observations
---

Workflow for combining Australian Gridded Climate Data NetCDF daily timeseries
into a single, unified NetCDF for later analysis for the Climate Health Reference 
Manual project.

Datasets to be combined should be located in a single folder, with no other
NetCDF (.nc) files present in the folder.

"""

import xarray as xr

# combine historical max temp gridded data
comb_xr_max = xr.open_mfdataset(
    'D:/Jake_ClimateRasters/historical_obs/tmax/*.nc',
    combine='by_coords',
    parallel=True
)

comb_xr_max = comb_xr_max.chunk({'time': 365, 'lat': 50, 'lon': 50})
comb_xr_max = comb_xr_max.rename({'tmax': 'tasmax'})
comb_xr_max.to_netcdf('D:/Jake_ClimateRasters/tasmax_1981_2010.nc', engine='h5netcdf')



# combine historical min temp gridded data
comb_xr_min = xr.open_mfdataset(
    'D:/Jake_ClimateRasters/historical_obs/tmin/*.nc',
    combine='by_coords',
    parallel=True
)

comb_xr_min = comb_xr_min.chunk({'time': 365, 'lat': 50, 'lon': 50})
comb_xr_min = comb_xr_min.rename({'tmin': 'tasmin'})
comb_xr_min.to_netcdf('D:/Jake_ClimateRasters/tasmin_1981_2010.nc', engine='h5netcdf')
