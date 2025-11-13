## SO Gaussian Foregrounds

This repo contains a script and a notebook to generate Gaussian foregrounds for Simons Observatory BB-forecasting. The notebook demonstrates the method with validation plots at each stage. The script takes the functions developed in the notebook to set up a class and a script which can produce the simulation. 

### Dependency

Requires `skytools` for emission law and unit conversion. (`conda install skytools`)

#### Running the script
```
python generate_gaussian_fg.py
```