from PIL import Image, ImageOps
from myterial import blue_grey, blue_grey_dark, salmon_light, salmon_darker
from rich import print
import sys
from pathlib import Path
import os 
from brainrender import Scene
from brainrender.actors import Points
import pandas as pd
sys.path.append("./")
from paper.figures import INSET, SILHOUETTE
import numpy as np


print("[bold red]Running: ", Path(__file__).name)

colors=['#40E0D0','#FFA500','#9F2B68']
# camera settings
cam = {
    "pos": (-16170, -7127, 31776),
    "viewup": (0, -1, 0),
    "clippingRange": (27548, 67414),
    "focalPoint": (7319, 2861, -3942),
    "distance": 43901,
}

outlier=np.array(['JPCM-02874_2', 'JPCM-02874_4', 'JPCM-02987_1', 'SNA-083481_3',
       'JPCM-02875_1', 'JPCM-02875_3', 'JPCM-02987_1', 'JPCM-02987_3',
       'SNA-083480_1', 'SNA-085070_2', 'SNA-042178_2'])

# create scene and edit root
scene = Scene(inset=INSET, screenshots_folder=r"D:\Reduced_Datasets_v1\figures\fig_1\Extended_Figure")
scene.root._needs_silhouette = True
scene.root._silhouette_kwargs["lw"] = 1
scene.root.alpha(0.2)


# load Data from mapped location
search_dir = r'Y:\CW_DATA\9\Data'
data_folder=r"D:\Reduced_Datasets_v1\animal_data\wt"

#files = os.listdir(search_dir)
spatival_shift = pd.read_csv(
    r"D:\Reduced_Datasets_v1\location_data\BorderRefinement\spatial_shift_v2.csv")

pop_stats_raw = pd.read_csv(
    r"D:\Reduced_Datasets_v1\figures\data\pop_stats_raw_2nd_derv_new_class.csv")
pop_stats = pop_stats_raw.loc[
            (~pop_stats_raw.animal_id.isin(outlier))]
# get probe locations
files = pop_stats.animal_id.unique()
print(len(files))
probes_locs = []
for sess in range(len(files))[:]:
    print(files[sess])
    try:
        search_file = os.path.join(
            data_folder,files[sess],
            'mapped_location_long_7.csv')
        probes_locs.append(pd.read_csv(search_file,index_col=0))
    except:
        search_file = os.path.join(
            data_folder,files[sess],
            'mapped_location_long_5.csv')
        probes_locs.append(pd.read_csv(search_file,index_col=0))
    shift = spatival_shift.loc[spatival_shift.animal_id==files[sess],'shift'].values




# get single probe tracks
for locs_tmp in probes_locs:

    locs_tmp.loc[:,'AP_template']=np.ceil((locs_tmp.loc[:,'AP'].values*-1+5400)).astype(int)
    locs_tmp.loc[:,'ML_template']=(np.ceil(((locs_tmp.loc[:,'ML'].values)) + 5700)).astype(int)
    locs_tmp.loc[:,'DV_template'] = (locs_tmp.DV.values).astype(int)
    locs_tmp.channel=locs_tmp.channel.values + shift
    locs_tmp = locs_tmp.loc[(locs_tmp.channel>=0)&(locs_tmp.channel<=384)]

    locs=locs_tmp.loc[:,['AP_template','DV_template','ML_template']].values
    regs_tmp = locs_tmp.loc[:,['ROI']].values
    #k = int(len(locs) / 374.0)
    k = int(len(locs))
    #print(k)
    spheres = Points(locs,
            #points[["ccf_ap", "ccf_dv", "ccf_lr"]].values,
            colors='grey',
            alpha=0,
            radius=10,
        )
    spheres = scene.add(spheres)

    VPL_points=[]
    PO_points =[]
    PoT_points=[]

    for i in range(k):
        
        points = locs[i : (i + 1)]
        regs = regs_tmp[i]#points.allen_ontology.values
        #print(points)
        # color based on if probes go through selected regions
        if "PoT" == regs[0]:
            PoT_points.append(points)
            color = colors[1]
            alpha = 1
            sil = 1
        elif "VPL" == regs[0]:
            VPL_points.append(points)
            color = colors[0]
            alpha = 1
            sil = 1
        elif "PO" == regs[0]:
            PO_points.append(points)
            color = colors[2]
            alpha = 1
            sil = 1
        else:
            continue
            color = 'black'
            alpha = 0
            sil = 0

    if len(PoT_points)>0:    # render channels as points
        spheres = Points(np.vstack((PoT_points)),
            #points[["ccf_ap", "ccf_dv", "ccf_lr"]].values,
            colors=colors[1],
            alpha=1,
            radius=20,
        )
        spheres = scene.add(spheres)

        if SILHOUETTE and sil:
            scene.add_silhouette(spheres, lw=sil)
    if len(VPL_points)>0:    # render channels as points
        spheres = Points(np.vstack((VPL_points)),
            #points[["ccf_ap", "ccf_dv", "ccf_lr"]].values,
            colors=colors[0],
            alpha=1,
            radius=20,
        )
        spheres = scene.add(spheres)

        if SILHOUETTE and sil:
            scene.add_silhouette(spheres, lw=sil)

    if len(PO_points)>0:    # render channels as points
        spheres = Points(np.vstack((PO_points)),
            #points[["ccf_ap", "ccf_dv", "ccf_lr"]].values,
            colors=colors[2],
            alpha=1,
            radius=20,
        )
        spheres = scene.add(spheres)

        if SILHOUETTE and sil:
            scene.add_silhouette(spheres, lw=sil)


# Add brain regions
VPL = scene.add_brain_region(
    "VPL",
    hemisphere="left",
    alpha=0.2,
    silhouette=False,
    color=colors[0],
)
PO = scene.add_brain_region(
    "PO",
    hemisphere="left",
    alpha=0.3,
    silhouette=False,
    color=colors[2],
)

PoT = scene.add_brain_region(
    "PoT",
    hemisphere="left",
    alpha=0.2,
    silhouette=False,
    color=colors[1],
)
'''th = scene.add_brain_region(
    "TH", alpha=0.1, silhouette=False, color='grey'
)
th.wireframe()'''
scene.add_silhouette(VPL,PO,PoT, lw=2)

# render
scene.render(zoom=3.5, camera='sagittal')
scene.screenshot(name="probes")
scene.close()

# Mirror image
#im = Image.open("paper/screenshots/probes.png")
#im_mirror = ImageOps.mirror(im)
#im_mirror.save("paper/screenshots/probes.png")