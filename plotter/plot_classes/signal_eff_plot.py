from puma.line_plot_2d import Line2D, Line2DPlot
from puma.metrics import eff_err
from plotter.config_dict import ConfigDict
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotter.plot_classes.plotbase import PlotBase

import matplotlib.pyplot as plt

def sci_notation_latex(x, precision=1):
    coeff = f"{x:.{precision}e}"
    base, exp = coeff.split("e")
    return rf"{base} \times 10^{{{int(exp):d}}}"

class SignalEffPlotBase(PlotBase):
    def plot(self):
        required_params = {
            "n_ratio_panels",
            "ylabel",
            "xlabel",
            "atlas_first_tag",
            "atlas_second_tag",
            "figsize",
            "y_scale",
            "label_fontsize",
            "fontsize",
            "atlas_fontsize",
            "grid"
        }
        filtered_params = {
            key: value for key, value in self.config.style.items() if key in required_params
        }
        signal_eff_plot = Line2DPlot(**filtered_params)

        all_lifetimes = {}
        all_signal_eff = {}
        all_signal_eff_err = {}

        for sample_name, sample in self.config.samples.items():
            sample_config = ConfigDict(sample)
            input_dir = sample_config.input_dir

            # store lifetimes and signal efficiencies for plotting
            lifetimes = []
            signal_eff = []
            signal_eff_err = []

            for fname in sorted(os.listdir(input_dir)):
                fpath = os.path.join(input_dir, fname)

                # extract lifetime from filename
                lifetime = fname.split("_")[-1].split(".")[0][1:]
                lifetimes.append(int(lifetime))
                # get efficiency
                with h5py.File(fpath, "r") as hdf_file:
                    
                    ds = hdf_file[sample_config.df_name]

                    target_label = self.config.target_label

                    # get attribute name for GNN ej score
                    keys_list = list(ds.dtype.fields.keys())

                    # search for which key contains the GNN signal discriminant
                    for i, key in enumerate(keys_list):
                        if "pdisp" in key and "GN3ej" in key: #need GN3ej or it will select salt_pdisp
                            pDisp = keys_list[i]
                            break

                    df = pd.DataFrame(
                        {
                            target_label: np.array(ds[target_label]).transpose(),
                            pDisp: np.array(ds[pDisp]).transpose(),
                        }
                    ).dropna()

                    # defining boolean array to select the different flavour classes
                    is_hs = df[target_label] == 1

                    # defining target efficiency
                    cut = self.config.cut_value

                    sig_disc = df[is_hs][pDisp]     # convenient to store signal discriminants
                    N_signal = len(sig_disc)

                    true_pos = sig_disc[sig_disc >= cut]    # determine the signal that passes the cut
                    eff = len(true_pos)/N_signal
                    err = eff_err(np.array([eff]), N_signal)[0] #eff_err expects np array

                    signal_eff.append(eff)
                    signal_eff_err.append(err)
        
            # plot signal efficiency as a function of lifetime
            sorted_indices = np.argsort(lifetimes) #sort lifetimes and signal efficiencies in increasing order
            lifetimes = np.array(lifetimes)[sorted_indices]
            signal_eff = np.array(signal_eff)[sorted_indices]
            signal_eff_err = np.array(signal_eff_err)[sorted_indices]

            all_lifetimes[sample_name] = lifetimes
            all_signal_eff[sample_name] = signal_eff
            all_signal_eff_err[sample_name] = signal_eff_err

            line = Line2D(x_values = lifetimes, y_values = signal_eff, marker = 'o', markersize = 4, label = sample_config.label)

            signal_eff_plot.add(line)   
        
        signal_eff_plot.draw()
        signal_eff_plot.axis_top.set_xscale("log")

        for sample_name in all_lifetimes:
            lifetimes = all_lifetimes[sample_name]
            signal_eff = all_signal_eff[sample_name]
            signal_eff_err = all_signal_eff_err[sample_name]

            for key, line in signal_eff_plot.plot_objects.items():
                if line.label == self.config.samples[sample_name]['label']:
                    line_obj = line
                    break
            colour = line_obj.colour

            signal_eff_plot.axis_top.errorbar(lifetimes, signal_eff, yerr=signal_eff_err, fmt='none', ecolor=colour, capsize=4, zorder=0)

        signal_eff_plot.savefig(self.config.file_name, transparent=False, dpi = 600)
