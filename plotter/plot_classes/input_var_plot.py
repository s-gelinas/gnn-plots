"""Produce histogram of each input variable for different jet types"""

import numpy as np
import pandas as pd
import h5py

from ftag import Flavours
from puma import Histogram, HistogramPlot
from puma.utils import get_dummy_2_taggers, get_good_linestyles
from plotter.config_dict import ConfigDict
from plotter.plot_classes.plotbase import PlotBase

class InputVarPlotBase(PlotBase):
	"""
	InputVarPlotBase is a subclass of PlotBase specializing in plotting histograms of input variables
	over different jet types.
	"""

	def plot(self):
		# required parameters for discriminant plot. Set in 'style' key in config
		required_params = {
			'logy',
			'figsize',
			'fontsize',
			'dpi',
			'use_atlas_tag',
			'atlas_second_tag'
		}

		# filter only the necessary parameters from the config file
		filtered_params = {
		    key: value for key, value in self.config.style.items() if key in required_params
		}

		# assume only one sample in the config file
		sample_key = list(self.config.samples.keys())[0]

		# extracting sample details and storing as a dictionary
		sample = ConfigDict(self.config.samples[sample_key])

		# extract output directory
		output_dir = self.config.output_dir

		# extracting data and processing it
		with h5py.File(sample.path, "r") as hdf_file:
			# load data sets
			jets_ds = hdf_file["jets"]
			tracks_ds = hdf_file["tracks"]

			# load "isDisplaced" flag to classify jets
			is_disp = jets_ds["isDisplaced"] == 1
			is_prompt = jets_ds["isDisplaced"] == 0
			
			# load variables to plot
			if "variables" not in self.config or not self.config.variables:
				variables = {}
				variables["jets"] = [
        			{key: key} for key, dt in jets_ds.dtype.fields.items()
        			if np.issubdtype(dt[0], np.floating) or np.issubdtype(dt[0], np.unsignedinteger)
					]
				variables["tracks"] = [
        			{key: key} for key, dt in tracks_ds.dtype.fields.items()
        			if np.issubdtype(dt[0], np.floating) or np.issubdtype(dt[0], np.unsignedinteger)
					]
			else:
				variables = self.config.variables

			# loop through variables to plot
			for ds_name, variables_list in variables.items(): #loop through "jets", "tracks"
				for variable_dict in variables_list: #loop through variables to plot
					for var, xlabel in variable_dict.items(): 
						if ds_name == "jets":
							ds = jets_ds
							var_data = ds[var]
							var_disp = var_data[is_disp]
							var_prompt = var_data[is_prompt]
						
						elif ds_name == "tracks":
							ds = tracks_ds
							var_data = ds[var]
							var_disp = var_data[is_disp]
							var_prompt = var_data[is_prompt]
						
							# convert to 1d
							var_disp = var_disp.ravel()
							var_prompt = var_prompt.ravel()

							# get rid of nan entries
							var_disp = var_disp[~np.isnan(var_disp)]
							var_prompt = var_prompt[~np.isnan(var_prompt)]

						min_val = min(var_disp.min(), var_prompt.min())
						max_val = max(var_disp.max(), var_prompt.max())
		    
						linestyles = get_good_linestyles()[:2]

						# initialize histogram plot
						plot_histo = HistogramPlot(
							bins=np.linspace(min_val,max_val,50),
							n_ratio_panels=0,
							xlabel = xlabel,
							ylabel="Normalized number of jets",
							logy=True,
							leg_ncol=1,
							**filtered_params
						)

						# add the histograms
						plot_histo.add(
							Histogram(
								var_disp,
								label='Emerging jets',
								colour=Flavours["bjets"].colour,
								linestyle=linestyles[0]
							),
							reference=False
						)
						plot_histo.add(
							Histogram(
								var_prompt,
								label="QCD jets",
								colour=Flavours["cjets"].colour,
								linestyle=linestyles[1],
							),
							reference=False
						)

						plot_histo.draw()
						plot_histo.savefig(f"{output_dir}/{ds_name}/{var}.png", transparent=False)