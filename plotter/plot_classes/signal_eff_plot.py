from puma.line_plot_2d import Line2D, Line2DPlot
from puma.metrics import calc_eff, calc_rej
from plotter.config_dict import ConfigDict
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotter.plot_classes.plotbase import PlotBase

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

        for _, sample in self.config.samples.items():
            sample_config = ConfigDict(sample)
            input_dir = sample_config.input_dir

            # store lifetimes and signal efficiencies for plotting
            lifetimes = []
            signal_eff = []

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
                        if "pdisp" in key and "GN3ej" in key: #need GN3ej or it will select salt_pdisp (what is that?)
                            pDisp = keys_list[i]
                            break

                    df = pd.DataFrame(
                        {
                            target_label: np.array(ds[target_label]).transpose(),
                            pDisp: np.array(ds[pDisp]).transpose(),
                        }
                    ).dropna()

                    #print(fpath, ds[pDisp])

                    # defining boolean array to select the different flavour classes
                    is_pu = df[target_label] == 0
                    is_hs = df[target_label] == 1

                    # defining target efficiency
                    sig_eff = np.linspace(*self.config.range)

                    n_pu = sum(is_pu)

                    rej = calc_rej(
                        df[is_hs][pDisp].values, df[is_pu][pDisp].values, sig_eff
                    )

                    cut = self.config.cut_value

                    sig_disc = df[is_hs][pDisp]     # convenient to store signal discriminants
                    N_signal = len(sig_disc)

                    true_pos = sig_disc[sig_disc >= cut]    # determine the signal that passes the cut
                    eff_ = len(true_pos)/N_signal
                    rej_ = calc_rej(
                        df[is_hs][pDisp].values, df[is_pu][pDisp].values, eff_
                    )
                    signal_eff.append(eff_)

        
            # plot signal efficiency as a function of lifetime
            sorted_indices = np.argsort(lifetimes) #sort lifetimes and signal efficiencies in increasing order
            lifetimes = np.array(lifetimes)[sorted_indices]
            signal_eff = np.array(signal_eff)[sorted_indices]

            print(signal_eff)

            line = Line2D(x_values = lifetimes, y_values = signal_eff, marker = 'o', label = sample_config.label)

            signal_eff_plot.add(line)   
        
        signal_eff_plot.draw()
        signal_eff_plot.axis_top.set_xscale("log")

        signal_eff_plot.savefig(self.config.file_name, transparent=False)
