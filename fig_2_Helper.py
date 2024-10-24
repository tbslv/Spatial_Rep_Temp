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

import numpy as np
from scipy.integrate import simps


import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class fig_2:

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


	# Define the surface function F(x, y)
	def F (x, y):
		return x**4 * y

	def load_data(self,raster=False):
		

		if raster:
			rasters_raw=merge_data(self.search_dir,self.files,'rasters.npy')

		else:
			rasters_raw=0
		
		
		return rasters_raw


	def init_allen_sdk(self):
		
		mcc = MouseConnectivityCache(10,manifest_file='connectivity/mouse_connectivity_manifest.json')

		#template, template_info = mcc.get_template_volume()
		#annot, annot_info = mcc.get_annotation_volume()
		structure_tree = pd.read_csv(self.structure_tree_path)


		#annotation_raw = np.load(self.annotation_raw_path)

		return mcc,structure_tree
	
	def prepare_masks(self):
		
		masks_s={}
		masks_h = {}
		masks_c={}
		for s in range(len(self.structures_location_masks))[:]:

			structure_selection = [self.structures_location_masks[s]]

			ids = self.structure_tree.loc[
				self.structure_tree.acronym==self.structures_location_masks[s],'id'].values
			mask, cm= self.mcc.get_structure_mask(ids)

			condensend_mask = mask[:,:,:].sum(axis=1)
			condensend_mask[condensend_mask>0]=1
			masks_h[self.structures_location_masks[s]]= condensend_mask


			condensend_mask = mask[:,:,:].sum(axis=2)
			condensend_mask[condensend_mask>0]=1
			masks_s[self.structures_location_masks[s]]= condensend_mask


			condensend_mask = mask[:,:,:].sum(axis=0)
			condensend_mask[condensend_mask>0]=1
			masks_c[self.structures_location_masks[s]]= condensend_mask
			
		masks_h['VB'] = masks_h['VPL']+masks_h['VPM']
		masks_h['VB'][masks_h['VB']>0]=1

		masks_h['PO_complex'] = masks_h['PO']+masks_h['PoT']
		masks_h['PO_complex'][masks_h['PO_complex']>0]=1

		masks_s['VB'] = masks_s['VPL']+masks_s['VPM']
		masks_s['VB'][masks_s['VB']>0]=1

		masks_s['PO_complex'] = masks_s['PO']+masks_s['PoT']
		masks_s['PO_complex'][masks_s['PO_complex']>0]=1

		masks_c['VB'] = masks_c['VPL']+masks_c['VPM']
		masks_c['VB'][masks_c['VB']>0]=1

		masks_c['PO_complex'] = masks_c['PO']+masks_c['PoT']
		masks_c['PO_complex'][masks_c['PO_complex']>0]=1

		return masks_h,masks_s,masks_c

	def plot_cell_location_2(self):
		numbers={}
		fig,ax_list = plt.subplots(2,1,figsize=(2.25,2.25))
		plt.subplots_adjust(wspace = -.10, hspace = -.10)

		for s in range(len(self.innerBorder))[:2]:
			pop_stats = self.pop_stats_raw.copy()

			structure_selection = [self.innerBorder[s]]

			print(s)
			ax=ax_list[s]
			
			if s == 0:
				ax.imshow(np.zeros(self.masks_s[self.innerBorder[s]].shape), cmap='binary')
				plot_outlines(self.masks_s[self.innerBorder[0]], lw=.5,
											  ax=ax,color=self.colors[0],label=self.innerBorder[s])
			else:
				ax.imshow(np.zeros(self.masks_s[self.innerBorder[s]].shape), cmap='binary')

				plot_outlines(self.masks_s[self.innerBorder[1]], lw=.5,
										  ax=ax,color=self.colors[1],label=self.innerBorder[s])
				plot_outlines(self.masks_s[self.innerBorder[2]], lw=.5,
										  ax=ax,color=self.colors[2],label=self.innerBorder[s])

			kws = {"s": .1, "facecolor": "none","edgecolor":"red", "linewidth": 10}
			data_tmp = pop_stats.loc[(pop_stats.stimtemp=='22.0')&
									 (pop_stats.quality=='good')&
									 (pop_stats.waveforms=='clean')&
									 (pop_stats.ROI.isin([self.innerBorder[s]]))
									&(~pop_stats.animal_id.isin(self.outlier)) 
									&(pop_stats.functional_classification=='putative_RS')]
			print(data_tmp.shape)
			numbers[self.innerBorder[s]] = data_tmp.shape[0]

			sns.scatterplot(data=data_tmp,x='AP_template',y='DV_template',
							marker='.',hue='ROI',palette=['lightgrey']*3,
			   ax=ax,legend=False,hue_order=self.innerBorder,edgecolor='black',
			   linewidth=0.1,**{'s':25})
			
			if s>0:

				data_tmp = pop_stats.loc[(pop_stats.stimtemp=='22.0')&
									 (pop_stats.quality=='good')&
									 (pop_stats.waveforms=='clean')&
										 (pop_stats.ROI.isin(['PO']))
										&(~pop_stats.animal_id.isin(self.outlier))
										 &(pop_stats.functional_classification=='putative_RS')]
				

				sns.scatterplot(data=data_tmp,x='AP_template',y='DV_template',
								marker='.',hue='ROI',palette=['lightgrey']*3,
				   ax=ax_list[s],legend=False,hue_order=self.innerBorder,
								edgecolor='black',linewidth=0.1,**{'s':25})
			numbers['PO'] = data_tmp.shape[0]
			
		ax_list[0].text(580,350,'{}'.format(
			list(numbers.keys())[0]),fontsize=7)
		ax_list[1].text(650,300,'{}'.format(
			list(numbers.keys())[1]),fontsize=7)
		ax_list[1].text(800,300,'{}'.format(
			list(numbers.keys())[2]),fontsize=7)

		
		ax_list[0].set_xlim(550,800)
		ax_list[0].set_ylim(550,300)

		ax_list[1].set_ylim(550,300)
		ax_list[1].set_xlim(650,900)


		ax_list[0].set_ylabel('',rotation=0,labelpad=15,fontsize=7)
		ax_list[1].set_ylabel('',rotation=0,labelpad=15)

		#ax_list[0].set_title('sagittal')

		[ax_list[i].set_xlabel('')for i in range(2)]

		#[ax_list[i].set_ylabel('')for i in range(2)]

		[cleanAxes(ax_list[i],total=True) for i in range(2)]
		[cleanAxes(ax_list[i],total=True) for i in range(2)]

		ax_list[1].hlines(y=490, xmin=750, xmax=850, linewidth=2, color='k')
		#ax_list[0].hlines(y=330, xmin=700, xmax=800, linewidth=2, color='k')

		plt.tight_layout()
		#plt.savefig(r'D:\Reduced_Datasets_v1\figures\fig_1\cell_location_only_sagittal_NEW_5.svg',format='svg')
		plt.show()

		return numbers 
	


	def plot_dist(xx, yy, f_cold_norm, counts, outlines_large, positions_margx,
						 f_cold_margx_norm, positions_margy, f_cold_margy_norm, params,
						 color_map,max_val,color,levs,x1,x2,x_steps,y1,y2,y_steps,shift):
		"""
		Plots the cold VPL distribution with contours and marginal distributions along the x and y axes.
		
		Parameters:
		- xx, yy: Meshgrid arrays representing x and y coordinates.
		- f_cold_norm: Normalized 2D cold distribution KDE.
		- counts: Array representing the number of counts for contour plotting.
		- outlines_large: Array defining the boundaries of the region of interest (ROI).
		- positions_margx: Marginal positions along the x-axis.
		- f_cold_margx_norm: Normalized marginal cold distribution along the x-axis.
		- positions_margy: Marginal positions along the y-axis.
		- f_cold_margy_norm: Normalized marginal cold distribution along the y-axis.
		- params: Parameter object containing colors and other properties.
		
		Returns:
		- None
		"""
		fig = plt.figure(figsize=(1.5, 1.5))

		# Define the axes positions
		ax_margx = placeAxesOnGrid(fig, xspan=[0, 0.8], yspan=[0, 0.15])
		ax_margy = placeAxesOnGrid(fig, xspan=[0.85, 1], yspan=[0.2, 1])
		ax = placeAxesOnGrid(fig, xspan=[0, 0.8], yspan=[0.2, 1])

		# Plot the outlines of the region of interest (ROI)
		cl = LineCollection([outlines_large], color='black', lw=0.5)
		ax.add_collection(cl)

		# Plot the filled contour plot for the cold distribution
		cfset = ax.contourf(xx, yy, f_cold_norm, cmap=color_map, levels=np.linspace(0, .1+max_val, levs), alpha=1)
		ax.contour(xx, yy, counts, levels=[1, 100])

		# Set x and y limits, ticks, and labels
		ax.set_xlim(x1, x2)
		ax.set_xticks(np.linspace(x1, x2, x_steps))
		ax.set_xticklabels(np.linspace(x1, x2, x_steps) / 1000)

		ax.set_yticks(np.linspace(y1, y2, y_steps))
		ax.set_yticklabels(np.linspace(-y1, -y2, y_steps) / 1000)
		ax.set_ylim(y2+shift, y1+shift)

		ax.set_xlabel('AP\n(mm)')
		ax.set_ylabel('DV\n(mm)')
		
		# Remove top and right spines
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		# Plot marginal distribution for x-axis
		ax_margx.plot(positions_margx, f_cold_margx_norm, color=color)
		ax_margx.fill_between(positions_margx, f_cold_margx_norm, color=color, alpha=0.5)
		ax_margx.set_xlim(x1, x2)
		ax_margx.set_yticks([0, max_val])
		ax_margx.set_yticklabels(['', str(int(max_val*100))])
		ax_margx.set_xticklabels([])
		ax_margx.set_ylabel('Prob.\n(%)')
		ax_margx.spines['right'].set_visible(False)
		ax_margx.spines['top'].set_visible(False)
		ax_margx.set_ylim(0, max_val)

		# Plot marginal distribution for y-axis
		ax_margy.plot(f_cold_margy_norm, positions_margy, color=color)
		ax_margy.fill_betweenx(positions_margy, f_cold_margy_norm, color=color, alpha=0.5)
		ax_margy.set_ylim(y2+shift, y1+shift)
		ax_margy.set_xticks([0, max_val])
		ax_margy.set_xticklabels(['', str(int(max_val*100))])
		ax_margy.set_yticklabels([])
		ax_margy.set_xlabel('Prob.\n(%)')
		ax_margy.set_xlim(0, max_val)

		# Remove top and right spines
		ax_margy.spines['right'].set_visible(False)
		ax_margy.spines['top'].set_visible(False)

		plt.show()


	def norm_dist(f_cold, f_null, f_cold_margx, f_null_margx, f_cold_margy,
					 f_null_margy, xx, yy, outlines_large, positions_margx, positions_margy, counts):
		"""
		Normalize the cold distribution (f_cold) with the null distribution (f_null), 
		apply boundary conditions based on outlines_large, and mask values based on thresholds.
		
		Parameters:
		- f_cold: 2D array of the cold distribution KDE.
		- f_null: 2D array of the null distribution KDE.
		- f_cold_margx: 1D array of the marginal distribution of cold data along the x-axis.
		- f_null_margx: 1D array of the marginal distribution of null data along the x-axis.
		- f_cold_margy: 1D array of the marginal distribution of cold data along the y-axis.
		- f_null_margy: 1D array of the marginal distribution of null data along the y-axis.
		- xx, yy: Meshgrid arrays representing x and y coordinates.
		- outlines_large: Array defining the boundaries of the region of interest (ROI).
		- positions_margx, positions_margy: Arrays of marginal positions for x and y axes.
		- counts: Array representing the number of counts for masking.

		Returns:
		- f_cold_norm: Normalized and masked 2D cold distribution.
		- f_cold_margx_norm: Normalized and masked marginal distribution along the x-axis.
		- f_cold_margy_norm: Normalized and masked marginal distribution along the y-axis.
		"""
		
		# Normalize the cold distribution with the null distribution
		f_cold_norm = f_cold / f_null
		f_cold_margx_norm = f_cold_margx / f_null_margx
		f_cold_margy_norm = f_cold_margy / f_null_margy

		
		# Apply boundary conditions for the 2D distribution
		f_cold_norm[np.where(xx < outlines_large[:, 0].min())] = 0
		f_cold_norm[np.where(xx > outlines_large[:, 0].max())] = 0
		f_cold_norm[np.where(yy < outlines_large[:, 1].min())] = 0
		f_cold_norm[np.where(yy > outlines_large[:, 1].max())] = 0
		
		# Apply boundary conditions for marginal distributions (x-axis)
		f_cold_margx_norm[np.where(positions_margx < outlines_large[:, 0].min())] = 0
		f_cold_margx_norm[np.where(positions_margx > outlines_large[:, 0].max())] = 0
		
		# Apply boundary conditions for marginal distributions (y-axis)
		f_cold_margy_norm[np.where(positions_margy < outlines_large[:, 1].min())] = 0
		f_cold_margy_norm[np.where(positions_margy > outlines_large[:, 1].max())] = 0
		
		# Mask based on counts for x-axis
		xmask_min = np.where(counts > 1)[0].min()
		xmask_max = np.where(counts > 1)[0].max()
		
		f_cold_norm[:xmask_min, :] = 0
		f_cold_norm[xmask_max:, :] = 0

		f_cold_margx_norm[:xmask_min] = 0
		f_cold_margx_norm[xmask_max:] = 0
		
		# Mask based on counts for y-axis
		ymask_min = np.where(counts > 1)[1].min()
		ymask_max = np.where(counts > 1)[1].max()
		
		f_cold_norm[:, :ymask_min] = 0
		f_cold_norm[:, ymask_max:] = 0

		f_cold_margy_norm[:ymask_min] = 0
		f_cold_margy_norm[ymask_max:] = 0
		
		return f_cold_norm, f_cold_margx_norm, f_cold_margy_norm


	def compute_dist(pop_stats_all_2,ROIs,stimulus, xx, positions, positions_margx, positions_margy, bw_method=0.5):
		"""
		Compute the cold distribution for the VPL region based on AP and DV coordinates.
		The function uses a Gaussian kernel density estimation (KDE) to generate 2D and marginal distributions.

		Parameters:
		- pop_stats_all_cold: DataFrame containing AP and DV coordinates for cold VPL data.
		- xx: Grid coordinates used for the 2D KDE.
		- positions: Flattened grid coordinates for KDE evaluation.
		- positions_margx: Marginal positions along the x-axis (AP).
		- positions_margy: Marginal positions along the y-axis (DV).
		- bw_method: Bandwidth method for KDE (default: 0.5).

		Returns:
		- f_cold: 2D kernel density estimate of the cold distribution.
		- f_cold_margx: Marginal distribution along the x-axis (AP).
		- f_cold_margy: Marginal distribution along the y-axis (DV).
		"""
		
		# Extract AP and DV coordinates for cold distribution

		dist=pop_stats_all_2.loc[(pop_stats_all_2.ROI.isin(ROIs))
				   &(pop_stats_all_2.sig==stimulus),['AP','DV']].values


		
		x_dist = dist[:, 0]
		y_dist = dist[:, 1]

		# Compute 2D KDE for the distribution (cold/warm)
		values_dist = np.vstack([x_dist, y_dist])
		kernel_dist = st.gaussian_kde(values_dist, bw_method=bw_method)
		f_dist = np.reshape(kernel_dist(positions).T, xx.shape)
		f_dist = f_dist * dist.shape[0]  # Scale KDE by number of points

		# Compute marginal KDE for x-axis (AP)
		kernel_margx = st.gaussian_kde(x_dist, bw_method=bw_method)
		f_margx = kernel_margx(positions_margx)
		f_margx = f_margx * dist.shape[0]  # Scale by number of points

		# Compute marginal KDE for y-axis (DV)
		kernel_margy = st.gaussian_kde(y_dist, bw_method=bw_method)
		f_margy = kernel_margy(positions_margy)
		f_margy = f_margy * dist.shape[0]  # Scale by number of points

		return f_dist, f_margx, f_margy



	def integrate_2D_surface(F, x_min=0, x_max=1, y_min=0, y_max=5, n_points_x=50, n_points_y=50,
							 f_null=None, positions_margx=None, positions_margy=None):
		"""
		Function to perform 2D integration of a surface using Simpson's rule.

		Parameters:
		- F: Function to integrate. Should accept two variables, x and y.
		- x_min, x_max: The range of x values (default: 0 to 1).
		- y_min, y_max: The range of y values (default: 0 to 5).
		- n_points_x: Number of points along the x-axis (default: 50).
		- n_points_y: Number of points along the y-axis (default: 50).
		- f_null: 2D array representing function values to integrate over (optional).
		- positions_margx: Marginal positions along the x-axis (optional).
		- positions_margy: Marginal positions along the y-axis (optional).
		
		Returns:
		- counts: Integrated values over small 2x2 windows using Simpson's rule.
		"""

		# Generate x and y points
		x = np.linspace(x_min, x_max, n_points_x)
		y = np.linspace(y_min, y_max, n_points_y)

		# Compute the function over the grid, using broadcasting
		zz = F(x.reshape(-1, 1), y.reshape(1, -1))

		# Perform the 2D integration using Simpson's rule
		integral_result = simps([simps(zz_x, x) for zz_x in zz], y)

		# If f_null, positions_margx, and positions_margy are provided, calculate the counts
		if f_null is not None and positions_margx is not None and positions_margy is not None:
			counts = np.zeros(f_null.shape)
			
			# Loop through grid and integrate small 2x2 sections
			for i in range(positions_margx.shape[0] - 2):
				for j in range(positions_margx.shape[0] - 2):
					counts[i, j] = simps(
						[simps(f_null[i:int(i+2), j:int(j+2)][k], positions_margx[i:int(i+2)])
						 for k in range(2)], 
						positions_margy[j:int(j+2)]
					)
		else:
			counts = None

		return integral_result, counts

	# Example usage of the function:



	def compute_null_dist(params, outlines,ROIs):

		import scipy.stats as st
		"""
		Function to compute the null distribution for VPL region using AP and DV coordinates 
		and generate KDE and marginal distributions.
		

		
		Returns:
		- f_null: 2D array representing the kernel density estimation (KDE)
		- f_null_margx: Marginal distribution along the x-axis (AP)
		- f_null_margy: Marginal distribution along the y-axis (DV)
		- xx, yy: Grid coordinates used for the 2D KDE
		"""
		
		# Extract relevant data for null distribution based on filtering criteria
		null_dist = params.pop_stats_raw.loc[
				#(params.pop_stats_raw.responsive_ex_stim==True)
				#(params.pop_stats_raw.ctype2.isin(['cold','cold/warm','warm']))
				(params.pop_stats_raw.stimtemp.isin(['22.0']))
				&(params.pop_stats_raw.ROI.isin(ROIs))
				&(params.pop_stats_raw.waveforms=='clean')
				&(params.pop_stats_raw.quality=='good')
				&(~params.pop_stats_raw.animal_id.isin(params.outlier))
				#&(params.pop_stats_raw.move_sig=='0')
				&(params.pop_stats_raw.functional_classification=='putative_RS'),
				['AP','DV']].values

		x = null_dist[:, 0]  # AP coordinates
		y = null_dist[:, 1]  # DV coordinates

		# Define the borders for the grid
		deltaX = (max(x) - min(x)) / 5
		deltaY = (max(y) - min(y)) / 5

		# Rescale the outlines
		outlines_large = np.zeros(outlines[0].shape)
		outlines_large[:, 0] = 5400 - outlines[0][:, 0] * 10
		outlines_large[:, 1] = outlines[0][:, 1] * 10

		xmin = min(x) - deltaX
		xmax = max(x) + deltaX
		ymin = min(y) - deltaY
		ymax = max(y) + deltaY

		# Create grid for 2D KDE
		xx, yy = np.mgrid[xmin:xmax:30j, ymin:ymax:30j]
		positions = np.vstack([xx.ravel(), yy.ravel()])
		values = np.vstack([x, y])

		# Perform kernel density estimation
		kernel = st.gaussian_kde(values, bw_method=.5)
		f_null = np.reshape(kernel(positions).T, xx.shape)
		f_null = f_null * null_dist.shape[0]  # Scale KDE by the number of points

		# Marginal distribution along x-axis (AP)
		positions_margx = np.linspace(xmin, xmax, 30)
		kernel_margx = st.gaussian_kde(x, bw_method=.5)
		f_null_margx = kernel_margx(positions_margx)
		f_null_margx = f_null_margx * null_dist.shape[0]

		# Marginal distribution along y-axis (DV)
		positions_margy = np.linspace(ymin, ymax, 30)
		kernel_margy = st.gaussian_kde(y, bw_method=.5)
		f_null_margy = kernel_margy(positions_margy)
		f_null_margy = f_null_margy * null_dist.shape[0]

		return f_null, f_null_margx, f_null_margy, xx, yy, positions_margx, positions_margy,kernel_margx,kernel_margy,positions

	def plot_outlines(bool_img, ax=None, **kwargs):
		if ax is None:
			ax = plt.gca()

		edges = get_all_edges(bool_img=bool_img)
		edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
		outlines = close_loop_edges(edges=edges)
		cl = LineCollection(outlines, **kwargs)
		ax.add_collection(cl)

		return outlines



	import pandas as pd

	def make_data(self, region=['VPL']):
		"""
		Create a combined dataset of cold and warm signals for a specified region (e.g., VPL).

		Parameters:
		- self: A dictionary-like object containing the raw data (`pop_stats_raw`) and any additional parameters such as outliers.
		- signal_types: List of signal types to include (default is ['cold', 'warm']). You can also pass ['cold/warm'] if needed.
		- region: The region of interest (default is 'VPL').
		- stimtemp_cold: Stimulation temperature for cold signals (default is '22.0').
		- stimtemp_warm: Stimulation temperature for warm signals (default is '42.0').

		Returns:
		- pop_stats_all_2: A concatenated DataFrame containing both cold and warm signals filtered by region and other parameters.
		"""
		
		# Cold signal dataset

		pop_stats_all_cold = self.pop_stats_raw.loc[
				(self.pop_stats_raw.ctype2.isin(['cold', 'cold/warm'])) &
				(self.pop_stats_raw.stimtemp == '22.0') &
				(self.pop_stats_raw.ROI.isin(region)) &
				(self.pop_stats_raw.waveforms == 'clean') &
				(self.pop_stats_raw.quality == 'good') &
				(~self.pop_stats_raw.animal_id.isin(self.outlier)) &
				(self.pop_stats_raw.functional_classification == 'putative_RS')
			]
		pop_stats_all_cold.loc[:, 'sig'] = 'cold'

		# Warm signal dataset

		pop_stats_all_warm = self.pop_stats_raw.loc[
				(self.pop_stats_raw.ctype2.isin(['warm', 'cold/warm'])) &
				(self.pop_stats_raw.stimtemp == '42.0') &
				(self.pop_stats_raw.ROI.isin(region)) &
				(self.pop_stats_raw.waveforms == 'clean') &
				(self.pop_stats_raw.quality == 'good') &
				(~self.pop_stats_raw.animal_id.isin(self.outlier)) &
				(self.pop_stats_raw.functional_classification == 'putative_RS')
			]
		pop_stats_all_warm.loc[:, 'sig'] = 'warm'

		# Combine cold and warm data into a single DataFrame
		pop_stats_all_2 = pd.concat((pop_stats_all_cold, pop_stats_all_warm), axis=0)
		
		return pop_stats_all_2

	import pandas as pd

	def load_anatomy(file_path, vpl_border=-1350, po_border=-1800):
	    """
	    Load and process anatomical data for thalamic cells from a given CSV file.
	    The function categorizes cells into subregions based on their structure and AP coordinate.

	    Parameters:
	    - file_path: Path to the CSV file containing the anatomical data.
	    - vpl_border: Anterior-posterior (AP) border separating aVPL from pVPL (default is -1350).
	    - po_border: AP border separating aPO from pPO (default is -1800).

	    Returns:
	    - thalamic_cells: DataFrame with added 'AP' and 'subregion' columns.
	    """
	    
	    # Load the CSV file
	    tracing_cells = pd.read_csv(file_path, index_col=0)

	    # Filter for thalamic cells in VPL, PO, and PoT regions
	    thalamic_cells = tracing_cells.loc[tracing_cells.structure.isin(['VPL', 'PO', 'PoT'])]

	    # Convert x-coordinates to AP by subtracting from 5400
	    thalamic_cells.loc[:, 'AP'] = 5400 - thalamic_cells.x.values

	    # Initialize subregion column with "not_assigned"
	    thalamic_cells.loc[:, 'subregion'] = "not_assigned"

	    # Assign subregions based on AP coordinates and structure
	    thalamic_cells.loc[(thalamic_cells.AP >= vpl_border) & (thalamic_cells.structure == 'VPL'), 'subregion'] = "aVPL"
	    thalamic_cells.loc[(thalamic_cells.AP < vpl_border) & (thalamic_cells.structure == 'VPL'), 'subregion'] = "pVPL"
	    
	    thalamic_cells.loc[(thalamic_cells.AP >= po_border) & (thalamic_cells.structure == 'PO'), 'subregion'] = "aPO"
	    thalamic_cells.loc[(thalamic_cells.AP < po_border) & (thalamic_cells.structure == 'PO'), 'subregion'] = "pPO"

	    # Assign subregion for PoT cells
	    thalamic_cells.loc[thalamic_cells.structure == 'PoT', 'subregion'] = "PoT"

	    return thalamic_cells


	def plot_retrolabeled_cells(thalamic_cells, order=['aVPL', 'pVPL', 'aPO', 'pPO', 'PoT'], 
	                           hue_order=['S1', 'IC'], palette=['lightgrey', 'black'], save_path=None):
	    """
	    Plot the proportion of retrogradely labeled cells across thalamic subregions based on their projections.

	    Parameters:
	    - thalamic_cells: DataFrame containing the thalamic cell data with 'subregion' and 'projection' columns.
	    - order: List specifying the order of the subregions on the x-axis (default: ['aVPL', 'pVPL', 'aPO', 'pPO', 'PoT']).
	    - hue_order: List specifying the order of projection types for the hue (default: ['S1', 'IC']).
	    - palette: List specifying the color palette for the projections (default: ['lightgrey', 'black']).
	    - save_path: Path to save the figure (optional). If provided, the plot will be saved as an SVG file.

	    Returns:
	    - fig, ax: Matplotlib figure and axis objects for further customization if needed.
	    """
	    
	    # Create the figure and axis
	    fig, ax = plt.subplots(figsize=(1.5, 0.75))
	    thalamic_cells['subregion'] = pd.Categorical(thalamic_cells['subregion'], categories=order, ordered=True)
	    # Plot using Seaborn's histplot with multiple='fill' to show proportions
	    sns.histplot(data=thalamic_cells, x='subregion', hue='projection', multiple='fill', 
	                 ax=ax, palette=palette, hue_order=hue_order, lw=0.5, shrink=0.75)

	    # Remove the legend
	    ax.legend([])

	    # Set y-axis ticks and labels to show proportion percentages
	    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
	    ax.set_yticklabels([0, 25, 50, 75, 100])

	    # Set x-axis limits and labels
	    ax.set_xlim(-0.5, len(order) - 0.5)

	    # Remove x-axis label and set y-axis label
	    ax.set_xlabel('')
	    ax.set_ylabel('Proportion (%)')

	    # Adjust visual style (remove top and right spines)
	    sns.despine()

	    # Optionally save the plot
	    if save_path:
	        plt.savefig(save_path, format='svg')

	    # Show the plot
	    plt.show()

	    return fig, ax

	def make_data_stats_ROI(self):

	    pop_stats_all_cold = self.pop_stats_raw.loc[
	                #(self.pop_stats_raw.responsive_ex_stim==True)
	                (self.pop_stats_raw.ctype2.isin(['cold','cold/warm','warm'
	                                                         ]))
	                &(self.pop_stats_raw.stimtemp.isin(['22.0']))
	                &(self.pop_stats_raw.ROI.isin(['PO','PoT','VPL']))
	                &(self.pop_stats_raw.waveforms=='clean')
	                &(self.pop_stats_raw.quality=='good')
	                &(~self.pop_stats_raw.animal_id.isin(self.outlier))
	                #&(self.pop_stats_raw.move_sig=='0')
	                &(self.pop_stats_raw.functional_classification=='putative_RS')]
	    pop_stats_all_warm = self.pop_stats_raw.loc[
	                #(self.pop_stats_raw.responsive_ex_stim==True)
	                (self.pop_stats_raw.ctype2.isin(['cold/warm','warm','cold']))
	                &(self.pop_stats_raw.stimtemp.isin(['42.0']))
	                &(self.pop_stats_raw.ROI.isin(['PO','PoT','VPL']))
	                &(self.pop_stats_raw.waveforms=='clean')
	                &(self.pop_stats_raw.quality=='good')
	                &(~self.pop_stats_raw.animal_id.isin(self.outlier))
	                #&(self.pop_stats_raw.move_sig=='0')
	                &(self.pop_stats_raw.functional_classification=='putative_RS')]

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

	    data.loc[:,'base_corr'] = data.peak_fr_stim - data.peak_fr_base

	    return data



	def plot_resptype_fractions(data, order=['aVPL', 'pVPL', 'aPO', 'pPO', 'PoT'], 
	                                hue_order=['cold', 'cold/warm', 'warm'], 
	                                
	                                stim_temp='22.0', roi_list=['VPL', 'PO', 'PoT'], save_path=None):
	    """
	    Plot the proportion of stimulus temperature types (cold, cold/warm, warm) across thalamic subregions.

	    Parameters:
	    - data: DataFrame containing data with 'subregion', 'stimtemp', 'ROI', and 'new_type' columns.
	    - order: List specifying the order of the subregions on the x-axis (default: ['aVPL', 'pVPL', 'aPO', 'pPO', 'PoT']).
	    - hue_order: List specifying the order of temperature types for the hue (default: ['cold', 'cold/warm', 'warm']).
	    - palette: List specifying the color palette for the temperature types (default: [params.blue, '#AD7748', params.red]).
	    - stim_temp: Temperature condition for filtering (default: '22.0').
	    - roi_list: List specifying which ROIs to include (default: ['VPL', 'PO', 'PoT']).
	    - save_path: Path to save the figure (optional). If provided, the plot will be saved as an SVG file.

	    Returns:
	    - fig, ax: Matplotlib figure and axis objects for further customization if needed.
	    """
	    palette=[self.blue, '#AD7748', self.red],
	    # Ensure 'subregion' follows the specified order
	    data['subregion'] = pd.Categorical(data['subregion'], categories=order, ordered=True)

	    # Filter the data based on stim_temp and ROI
	    filtered_data = data.loc[(data.stimtemp == stim_temp) & (data.ROI.isin(roi_list)) & 
	                               (data.new_type.isin(hue_order))]

	    # Create the figure and axis
	    fig,ax=plt.subplots(figsize=(1.5,0.75))
	    sns.histplot(data=data.loc[(data.stimtemp == stim_temp)
	                              &(data.ROI.isin(['VPL','PO','PoT']))
	                              &(data.new_type.isin(roi_list))
	                              ],x='subregion',hue='new_type',ax=ax,palette=palette,multiple='fill',
	                 hue_order=hue_order,lw=0.5,shrink=0.75)
	    ax.legend([])
	    #ax.set_xticklabels(['aVPL','pVPL','aPO','pPO','PoT'])
	    #ax.set_yscale('log')
	    #ax.set_xticks([str(0),str(1),str(2),str(3),str(4)])
	    ax.set_yticks([0,0.25,.5,.75,1])
	    ax.set_yticklabels([0,25,50,75,100])
	    ax.set_xlim(-.5,4.5)
	    ax.set_xlabel('')
	    ax.set_ylabel('Proportion (%)')
	    sns.despine()

	    # Optionally save the plot
	    if save_path:
	        plt.savefig(save_path, format='svg')

	    # Show the plot
	    plt.show()

	    return fig, ax

