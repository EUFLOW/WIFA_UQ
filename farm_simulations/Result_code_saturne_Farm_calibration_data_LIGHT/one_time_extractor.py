from netCDF4 import Dataset
import numpy as np

# Charger le fichier NetCDF d'origine
file_path = '/home/j02084/Simu/WP4/cs_run_1WT-107_newProfil/flow_field_interpolated.nc'  # Remplacez par le chemin de votre fichier
nc_orig = Dataset(file_path, 'r')

# Spécifier l'indice de temps que vous souhaitez extraire
ti = 15  # Remplacez par l'indice de temps souhaité

# Créer un nouveau fichier NetCDF
output_file_path = 'filtered_file.nc'  # Remplacez par le chemin de sortie souhaité
nc_new = Dataset(output_file_path, 'w', format='NETCDF4')

# Définir les dimensions
nc_new.createDimension('x', len(nc_orig.variables['x']))
nc_new.createDimension('y', len(nc_orig.variables['y']))
nc_new.createDimension('altitudes', len(nc_orig.variables['z']))
nc_new.createDimension('time', 1)

# Créer les variables
x = nc_new.createVariable('x', 'f8', ('x',))
y = nc_new.createVariable('y', 'f8', ('y',))
z = nc_new.createVariable('z', 'f8', ('altitudes',))
time = nc_new.createVariable('time', 'f8', ('time',))
wind_speed = nc_new.createVariable('wind_speed', 'f8', ('time', 'altitudes', 'x', 'y'))
wind_direction = nc_new.createVariable('wind_direction', 'f8', ('time', 'altitudes', 'x', 'y'))

# Copier les données
x[:] = nc_orig.variables['x'][:]
y[:] = nc_orig.variables['y'][:]
z[:] = nc_orig.variables['z'][:]
time[:] = nc_orig.variables['time'][ti]
wind_speed[:] = nc_orig.variables['wind_speed'][ti, :, :, :]
wind_direction[:] = nc_orig.variables['wind_direction'][ti, :, :, :]

# Fermer les fichiers
nc_orig.close()
nc_new.close()

print(f"Nouveau fichier NetCDF créé : {output_file_path}")
