from wifa.pywake_api import run_pywake
from windIO.utils.yml_utils import validate_yaml, load_yaml
dat = load_yaml('./windIO_1WT/wind_energy_system/system.yaml')
for kk in [.05, .1]:
    dat['attributes']['analysis']['wind_deficit_model']['wake_expansion_coefficient']['k_b'] = kk
    run_pywake(dat, output_dir='k_%.2f' % kk)

