import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#fpath = '/project/def-mdanning/sgelinas/salt/salt/logs/GN3ej-oldSamples-extLabels-fold0-classdict_20250610-T130836/ckpts/lifetime/epoch=011-val_loss=0.00779__test_pp_output_test_Py8EG_Zprime2EJs_Ld20_rho40_pi10_Zp1500_l1.h5'
fpath = '/project/def-mdanning/sgelinas/salt/salt/logs/GN3ej-combined-extLabels-fold0-classdict_20250606-T150049/ckpts/lifetime_epoch0/epoch=000-val_loss=0.03274__test_pp_output_test_Py8EG_Zprime2EJs_Ld20_rho40_pi10_Zp1500_l1.h5'
#fpath = '/project/def-mdanning/sgelinas/salt/salt/logs/GN3ej-combined-extLabels-fold0-classdict_20250606-T150049/ckpts/lifetime/epoch=007-val_loss=0.02522__test_pp_output_test_Py8EG_Zprime2EJs_Ld20_rho40_pi10_Zp1500_l1.h5'

with h5py.File(fpath, "r") as hdf_file:
    ds = hdf_file["jets"]
    keys_list = list(ds.dtype.fields.keys())
    for key in keys_list:
        print(key, ds[key][:5])