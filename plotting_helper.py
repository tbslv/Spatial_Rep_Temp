import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def plot_raster(data_tmp,ax1,trials,color='grey'):
	for i in range(trials):
		
		dots_x = (data_tmp[:,0][data_tmp[:,1]==i]).astype(int)
		dots = np.ones(dots_x.size)+i
		ax1.scatter(dots_x,dots,s=0.4,color=color)
	#cleanAxes(ax1,total=True)
	#plt.show()
	return
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def placeAxesOnGrid(fig,dim=[1,1],xspan=[0,1],yspan=[0,1],wspace=None,hspace=None,):
	'''
	Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec
	
	Takes as arguments:
		fig: figure handle - required
		dim: number of rows and columns in the subaxes - defaults to 1x1
		xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
		yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
		wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively
		
	returns:
		subaxes handles
		
	written by doug ollerenshaw
	'''
	outer_grid = gridspec.GridSpec(100,100)
	inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0],dim[1],
												  subplot_spec=outer_grid[int(100*yspan[0]):int(100*yspan[1]),int(100*xspan[0]):int(100*xspan[1])],
												  wspace=wspace, hspace=hspace)
	

	#NOTE: A cleaner way to do this is with list comprehension:
	# inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
	inner_ax = dim[0]*[dim[1]*[fig]] #filling the list with figure objects prevents an error when it they are later replaced by axis handles
	inner_ax = np.array(inner_ax)
	idx = 0
	for row in range(dim[0]):
		for col in range(dim[1]):
			inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx])
			fig.add_subplot(inner_ax[row,col])
			idx += 1

	inner_ax = np.array(inner_ax).squeeze().tolist() #remove redundant dimension
	return inner_ax


def cleanAxes(ax,bottomLabels=False,leftLabels=False,rightLabels=False,topLabels=False,total=False):
	ax.tick_params(axis='both',labelsize=10)
	ax.spines['top'].set_visible(False);
	ax.yaxis.set_ticks_position('left');
	ax.spines['right'].set_visible(False);
	ax.xaxis.set_ticks_position('bottom')
	if not bottomLabels or topLabels:
		ax.set_xticklabels([])
	if not leftLabels or rightLabels:
		ax.set_yticklabels([])
	if rightLabels:
		ax.spines['right'].set_visible(True);
		ax.spines['left'].set_visible(False);
		ax.yaxis.set_ticks_position('right');
	if total:
		ax.set_frame_on(False);
		ax.set_xticklabels('',visible=False);
		ax.set_xticks([]);
		ax.set_yticklabels('',visible=False);
		ax.set_yticks([])



def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels,bbox_to_anchor=new_loc, title=title, **kws)
    
from matplotlib.collections import LineCollection
def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            edges.append(np.array([[i, j+1],
                                   [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            edges.append(np.array([[i+1, j],
                                   [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            edges.append(np.array([[i, j],
                                   [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            edges.append(np.array([[i, j],
                                   [i, j+1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)


def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:

        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list


def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)


array = np.zeros((20, 20))
array[4:7, 3:8] = 1
array[4:7, 12:15] = 1
array[7:15, 7:15] = 1
array[12:14, 13:14] = 0



