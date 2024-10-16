import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import os 
from plotting_helper import placeAxesOnGrid, move_legend,cleanAxes

import seaborn as sns
import matplotlib
from matplotlib import rcParams
from data_IO import *
from plotting_helper import *
import matplotlib.ticker as ticker
import matplotlib as mpl
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler


from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns

class fig_1:

	def __init__(self):
		
		# define high-level varibales
		
		self.structure_tree_path=r"C:\Users\tobiasleva\Work\ExtracellularRecordings\SHARP-track\structure_tree_safe_2017.csv"
		self.annotation_raw_path = r"C:\Users\tobiasleva\Work\ExtracellularRecordings\SHARP-track\annotation_volume_10um_by_index.npy"
		self.base_folder =  r"D:\\Reduced_Datasets_v1"

		self.data_folder= os.path.join(self.base_folder,'animal_data','wt')
		self.spatial_shift = pd.read_csv(
		r"D:\Reduced_Datasets_v1\location_data\BorderRefinement\spatial_shift_v2.csv")

		self.outlier = np.array(['JPCM-02874_2', 'JPCM-02874_4', 'JPCM-02987_1',
								 'SNA-083481_3','JPCM-02875_1', 'JPCM-02875_3', 
								 'JPCM-02987_1', 'JPCM-02987_3','SNA-083480_1',
								  'SNA-085070_2', 'SNA-042178_2','SNA-042178_1',
								  'SNA-056315_1','SNA-056315_2'])

		self.search_dir=r'N:\CW_DATA\9\Data'
		self.files=os.listdir(self.search_dir)
		
		self.red=[0.8784,0.0863,0.1686]
		self.blue=[0,0.0032,0.6471]


		self.structures_location_masks = ['VPL','VPM','RT','PO','PoT','ZI']

		self.names=['Thalamus','root','ventricular systems','Cerebral cortex',
	   'Cerebral nuclei','Hippocampal formation','Striatum',
	   'fiber tracts','Brain stem','Isocortex','Hypothalamus','Midbrain']
	

		self.names_ROI = ['Ventral posterolateral nucleus of the thalamus',
			'Posterior complex of the thalamus',
			'Posterior triangular thalamic nucleus']

		self.innerBorder = ['VPL','PoT','PO']
		self.outer_border= ['VPL','PO_complex']

		self.colors=['#40E0D0','#FFA500','#9F2B68']


		self.binsize=.01
		self.sweeplength=20
		self.samplingrate=30000
		self.sigma = 15

	def load_data(self,raster=False):
		

		if raster:
			rasters_raw=merge_data(self.search_dir,self.files,'rasters.npy')

		else:
			rasters_raw=0
		
		
		return rasters_raw
	
	def hex_to_rgb(self, value):

		value = value.lstrip('#')
		lv = len(value)

		return tuple(int(value[i:i + lv // 3], 16)/255 for i in range(0, lv, lv // 3))

	def make_data_heatmaps(self,binsize=0.01, ROIS=['VPL','PO','PoT'], resptypes=['cold','cold/warm','warm'],
							baseline_start_time = 7, baseline_end_time = 9, functional_type = 'putative_RS',
							sigma = 15 ):
		
		pop_stats_selection_cold = self.pop_stats_raw.loc[
									(~self.pop_stats_raw.animal_id.isin(self.outlier))
									&(self.pop_stats_raw.ROI.isin(ROIS))
									&(self.pop_stats_raw.stimtemp=='22.0')
									&(self.pop_stats_raw.waveforms=='clean')
									&(self.pop_stats_raw.quality=='good')
								   &(self.pop_stats_raw.ctype2.isin(resptypes))
									&(self.pop_stats_raw.functional_classification==functional_type)
								   ].sort_values(['tuning_index'],ascending=False)

		baseline_start=int(baseline_start_time/binsize)
		baseline_end=int(baseline_end_time/binsize)
		end=int(self.sweeplength*self.samplingrate)
		binsize_sample=binsize*self.samplingrate
		bins=np.arange(0,end+binsize,binsize_sample).astype(int)
			
		data_cold = np.zeros((pop_stats_selection_cold.index.values.shape[0],bins.shape[0]))

		for idx,val  in enumerate(pop_stats_selection_cold.index.values):
			#val-=10
			spikes_tmp = np.sort(self.rasters_raw[val][:,0])
			spike_count,bins = np.histogram(spikes_tmp,bins=bins)

			data_cold[idx,:-1]=spike_count

		baselines = data_cold[:,baseline_start:baseline_end].mean(axis=1)

		data_cold_base=data_cold-baselines[:,np.newaxis]

		ticks_cold=pop_stats_selection_cold.combined_type.value_counts().values.cumsum()

		cold_ticks=np.where(pop_stats_selection_cold.responsive_ex_stim.values == 1)[0]


		pop_stats_selection_warm = self.pop_stats_raw.loc[
									(~self.pop_stats_raw.animal_id.isin(self.outlier))
									&(self.pop_stats_raw.ROI.isin(ROIS))
									&(self.pop_stats_raw.stimtemp=='42.0')
									&(self.pop_stats_raw.waveforms=='clean')
									&(self.pop_stats_raw.quality=='good')
								   &(self.pop_stats_raw.ctype2.isin(resptypes))
									&(self.pop_stats_raw.functional_classification==functional_type)
								   ].sort_values(['tuning_index'],ascending=False)


		data_warm = np.zeros((pop_stats_selection_warm.index.values.shape[0],bins.shape[0]))


		for idx,val  in enumerate(pop_stats_selection_warm.index.values):
			#val+=10
			spikes_tmp = np.sort(self.rasters_raw[val][:,0])
			spike_count,bins = np.histogram(spikes_tmp,bins=bins)

			data_warm[idx,:-1]=spike_count

		baselines = data_warm[:,baseline_start:baseline_end].mean(axis=1)


		data_warm_base=data_warm-baselines[:,np.newaxis]

		ticks_warm=pop_stats_selection_warm.combined_type.value_counts().values.cumsum()
		warm_ticks=np.where(pop_stats_selection_warm.responsive_ex_stim.values == 1)[0]


		colors_cold = pop_stats_selection_cold.ctype2.values
		colors_cold[colors_cold == 'warm'] = '#FFFFFF'
		colors_cold[colors_cold == 'cold/warm'] = '#AD7748'
		colors_cold[colors_cold == 'cold'] = '#0000A5'

		colors_warm= pop_stats_selection_warm.ctype2.values
		colors_warm[colors_warm == 'warm'] = '#DF162A'
		colors_warm[colors_warm == 'cold/warm'] = '#AD7748'
		colors_warm[colors_warm == 'cold'] = '#FFFFFF'


		data_warm_smooth =np.zeros(data_warm_base.shape)
		data_cold_smooth =np.zeros(data_warm_base.shape)

		for i in range(data_warm_smooth.shape[0]):
			data_warm_smooth[i,:] = gaussian_filter1d(data_warm_base[i,:], sigma)
			data_cold_smooth[i,:] = gaussian_filter1d(data_cold_base[i,:], sigma)

		data_tot = np.hstack((data_cold_smooth,data_warm_smooth))

		scaler = MinMaxScaler()

		data_norm = scaler.fit_transform(data_tot.T)
		data_norm = data_norm.T

		data_cold_smooth=data_norm[:,:int(self.sweeplength/binsize)]
		data_warm_smooth=data_norm[:,int(self.sweeplength/binsize):]
		
		return data_cold_smooth,colors_cold, data_warm_smooth,colors_warm


	def plot_heatmaps(self, cold_trace, warm_trace, data_cold_smooth, data_warm_smooth, colors_cold, colors_warm,
					 start_time=8,end_time=16,stim_start=9,stim_duration=2):
		
		fig = plt.figure(figsize=(3, 5.2))
		# Create subplots for cold and warm conditions
		ax_temp_cold = placeAxesOnGrid(fig, xspan=[0, .4], yspan=[0, .1])
		ax_temp_warm = placeAxesOnGrid(fig, xspan=[0.45, .85], yspan=[0, .1])

		ax_cold = placeAxesOnGrid(fig, yspan=[.1, 1], xspan=[0, 0.4])
		ax_warm = placeAxesOnGrid(fig, yspan=[.1, 1], xspan=[0.45, 0.85])

		cax = placeAxesOnGrid(fig, yspan=[0.85, 1], xspan=[0.9, 0.925])
		cax_2 = placeAxesOnGrid(fig, yspan=[0.85, 1], xspan=[0.93, .955])

		# Plot warm temperature trace
		ax_temp_warm.plot(warm_trace, color=self.red)
		ax_temp_warm.set_xlim(start_time*1000, end_time*1000)
		cleanAxes(ax_temp_warm, total=True)

		# Plot heatmap for warm data
		im_warm = ax_warm.imshow(data_warm_smooth[:, int( start_time/ (self.binsize)):int(end_time / self.binsize)], aspect='auto',
								 cmap='Reds', vmin=0, vmax=1, interpolation='gaussian')

		ax_warm.set_xticks([])
		ax_warm.set_yticks(np.arange(data_warm_smooth.shape[0]) - 6)
		ax_warm.set_yticklabels(['-'] * data_warm_smooth.shape[0], color='grey', fontsize=15)

		ax_warm.yaxis.set_tick_params(pad=-3)
		ax_warm.yaxis.set_tick_params(width=0)

		# Set tick colors for warm data
		for ticklabel, tickcolor in zip(ax_warm.get_yticklabels(), colors_warm):
			ticklabel.set_color(fig_1.hex_to_rgb(self, tickcolor))

		ax_warm.axvline(int((stim_start-start_time) / self.binsize), color='black', lw=.5, ls=':', alpha=.8)
		ax_warm.axvline(int((stim_start-start_time+stim_duration) / self.binsize), color='black', lw=.5, ls=':', alpha=.8)

		# Plot cold temperature trace
		ax_temp_cold.plot(cold_trace, color=self.blue)
		ax_temp_cold.set_xlim(start_time*1000, end_time*1000)
		cleanAxes(ax_temp_cold, total=True)

		# Plot heatmap for cold data
		im_cold = ax_cold.imshow(data_cold_smooth[:, int(start_time / (self.binsize)):int(end_time / self.binsize)], aspect='auto',
								 vmin=0, vmax=1, cmap='Blues', interpolation='gaussian')

		ax_cold.set_xticks([])
		ax_cold.set_yticks(np.arange(data_cold_smooth.shape[0]) - 6)
		ax_cold.set_yticklabels(['-'] * data_cold_smooth.shape[0], color='grey', fontsize=15)
		ax_cold.yaxis.set_tick_params(pad=-3)
		ax_cold.yaxis.set_tick_params(width=0)

		# Set tick colors for cold data
		for ticklabel, tickcolor in zip(ax_cold.get_yticklabels(), colors_cold):
			ticklabel.set_color(fig_1.hex_to_rgb(self, tickcolor))

		ax_cold.axvline(int((stim_start-start_time) / self.binsize), color='black', lw=.5, ls=':', alpha=.8)
		ax_cold.axvline(int((stim_start-start_time+stim_duration) / self.binsize), color='black', lw=.5, ls=':', alpha=.8)

		# Add colorbars
		fig.colorbar(im_cold, cax=cax, orientation='vertical')
		fig.colorbar(im_warm, cax=cax_2, orientation='vertical')

		# Customize colorbar ticks and labels
		cax.set_yticks([0, 1])
		cax_2.set_yticks([0, 1])

		cax.set_yticklabels([
			])

		cax_2.set_ylabel('norm.\n Firing rate ')

		cax.tick_params (axis='both', which='major', labelsize=7)
		cax_2.tick_params (axis='both', which='major', labelsize=7)

		# Set limits for cold and warm temperature axes
		ax_temp_cold.set_ylim(21, 43)
		ax_temp_warm.set_ylim(21, 43)

		ax_temp_cold.set_title(f'all thalamic neurons (n={data_cold_smooth.shape[0]})', fontsize=7)

		sns.despine(fig)
		ax_cold.spines['bottom'].set_visible(False)
		ax_warm.spines['bottom'].set_visible(False)

		# Display the plot
		plt.show()


	def make_data_frs(self,ROIS=['VPL','PO','PoT'], resptypes=['cold','cold/warm','warm'],functional_type = 'putative_RS'):
		# Selection of cold stats
		pop_stats_all_cold = self.pop_stats_raw.loc[
			(self.pop_stats_raw.ctype2.isin(resptypes))
			& (self.pop_stats_raw.stimtemp.isin(['22.0']))
			& (self.pop_stats_raw.ROI.isin(ROIS))
			& (self.pop_stats_raw.waveforms == 'clean')
			& (self.pop_stats_raw.quality == 'good')
			& (~self.pop_stats_raw.animal_id.isin(self.outlier))
			& (self.pop_stats_raw.functional_classification == functional_type)
		]

		# Selection of warm stats
		pop_stats_all_warm = self.pop_stats_raw.loc[
			(self.pop_stats_raw.ctype2.isin(resptypes))
			& (self.pop_stats_raw.stimtemp.isin(['42.0']))
			& (self.pop_stats_raw.ROI.isin(ROIS))
			& (self.pop_stats_raw.waveforms == 'clean')
			& (self.pop_stats_raw.quality == 'good')
			& (~self.pop_stats_raw.animal_id.isin(self.outlier))
			& (self.pop_stats_raw.functional_classification == functional_type)
		]

		# Concatenating cold and warm data
		data = pd.concat((pop_stats_all_cold, pop_stats_all_warm))

		# Assigning new_type based on ctype2
		data.loc[data.ctype2.isin(['cold']), 'new_type'] = 'cold'
		data.loc[data.ctype2.isin(['cold/warm']), 'new_type'] = 'cold/warm'
		data.loc[data.ctype2.isin(['warm']), 'new_type'] = 'warm'

		# Calculating base_corr
		data['base_corr'] = data.peak_fr_stim - data.peak_fr_base

		return data  


	def make_data_fractions(self,ROIS=['VPL','PO','PoT'], resptypes=['cold','cold/warm','warm'],functional_type = 'putative_RS'):
		pop_stats_all_cold = self.pop_stats_raw.loc[
				(self.pop_stats_raw.ctype2.isin(resptypes))
				& (self.pop_stats_raw.stimtemp.isin(['22.0']))
				& (self.pop_stats_raw.ROI.isin(ROIS))
				& (self.pop_stats_raw.waveforms == 'clean')
				& (self.pop_stats_raw.quality == 'good')
				& (~self.pop_stats_raw.animal_id.isin(self.outlier))
				& (self.pop_stats_raw.functional_classification == functional_type)
			]

			# Selection of warm stats
		pop_stats_all_warm = self.pop_stats_raw.loc[
				(self.pop_stats_raw.ctype2.isin(resptypes))
				& (self.pop_stats_raw.stimtemp.isin(['42.0']))
				& (self.pop_stats_raw.ROI.isin(ROIS))
				& (self.pop_stats_raw.waveforms == 'clean')
				& (self.pop_stats_raw.quality == 'good')
				& (~self.pop_stats_raw.animal_id.isin(self.outlier))
				& (self.pop_stats_raw.functional_classification == functional_type)
			]

		#cold_idxs = pop_stats_all_cold.indexer.unique()

		#warm_tmp=self.pop_stats_raw.loc[(self.pop_stats_raw.stimtemp=='42.0')
								 #&(self.pop_stats_raw.indexer.isin(cold_idxs))]

		data=pd.concat((pop_stats_all_cold,pop_stats_all_warm))

		data.loc[data.ctype2.isin(['cold']),
						 'new_type'] = 'cold'

		data.loc[data.ctype2.isin(['cold/warm']),
						 'new_type'] = 'cold/warm'

		data.loc[data.ctype2.isin(['warm']),
						 'new_type'] = 'warm'
		return data
		