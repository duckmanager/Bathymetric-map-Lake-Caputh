from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm


DATA_FILE = "output/multibeam/unfiltered_data.csv"              # Datei mit den Messdaten
FILTER_CSV = "output/multibeam/interactive_error/interactive_error_points.csv"  # CSV für die Indizes der auszuschließenden Punkte


# check if csv with filtered indices already exists, if so, load csv's filter points with pre-selected points and output corrected DataFrame
if Path(FILTER_CSV).exists():
    bad = pd.read_csv(FILTER_CSV, index_col=0).index
    df = pd.read_csv(DATA_FILE)

    # save and remove edge points from dataframe  - not included in filtering
    boundary_points = df[df['file_id'] == "artificial_boundary_points"]
    
    # removed already marked faulty points and add boundary points again
    df_corrected = pd.concat([df.drop(bad), boundary_points], ignore_index=True)
    
    print(f"({len(bad)}) marked error points got loaded and removed from data")
    exit()

# If no earlier error correction exists:

# read survey data
df = pd.read_csv(DATA_FILE)
# save edge points seperatly
boundary_points = df[df['file_id'] == "artificial_boundary_points"]

# remove edge points for further processing
df = df[df['file_id'] != "artificial_boundary_points"]

bad_indices = []  # collection of faulty point indices 

# Iterate through all surveys showing progress with tqdm
for fid in tqdm(df['file_id'].unique(), desc="Messfahrten"): # iterate through each individual survey by file_id
    subdf = df[df['file_id'] == fid] # create temporary dataframe only containing the current survey data
    pts_all = []  # list of all points and their specific informations [Distance, depth, index, original color]
    fig, ax = plt.subplots(figsize=(10, 6)) # create matplot figure
    markers = {}  # dict for temporary saving selection of points
    beams = subdf['Beam_type'].unique() # determine all unique beam types in the survey
    colors = plt.cm.tab10(np.linspace(0, 1, len(beams))) # color palette with uniqe color for each beam type
    
    for color, beam in zip(colors, beams): # iterate through each beam
        beam_df = subdf[subdf['Beam_type'] == beam].sort_index() # creates new dataframe for each beamkeeping original indices
        if beam_df.empty:
            continue
        x_coords = beam_df['Longitude'].values # saves x and y coordinates in UTM32N
        y_coords = beam_df['Latitude'].values
        # Berechnung der kumulativen Distanz (in Metern)
        dist = np.r_[0, np.cumsum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))] # calculates the direct distance between each point starting from 0 and taking the sum for each new one. - same procedure could be used for point interpolation
        depth = beam_df['Depth (m)'].values # saves eah depth value
        ax.scatter(dist, depth, label=beam, color=color) # plots scatterplot with dist on the x axis and depth on the y axis, beam type in the legend and specific color per beam
        for d, dep, idx in zip(dist, depth, beam_df.index): 
            pts_all.append((d, dep, idx, color))   # saves distance, depth, index and color of each point for later selection
    
    ax.set_xlabel("Fahrstrecke (m)") # x-axis label
    ax.set_ylabel("Tiefe (m)") # y-axis label
    if pts_all:
        min_depth = min(p[1] for p in pts_all) # searches smalles depth value
        # Y-Achse: oberster Wert 0, unterster Wert = minimaler Tiefenwert minus 0.5 m Puffer
        ax.set_ylim(min_depth - 0.5, 0) # set y-axis from 0 to lowest depth -0.5m for better visability
    ax.legend()
    plt.title(f"Messfahrt: {fid}")
    
    # Click-Event: sselects closest point (in display / pixel coordinates)
    def on_click(event): # when matplot recognizes a click
        if event.inaxes != ax: # return if click is out of the plot axis
            return
        if event.xdata is None or event.ydata is None: # return if click is out of plot area
            return
        click_disp = ax.transData.transform((event.xdata, event.ydata)) # matplot saves click coordinates in x and y of the plot - ax.transData.transform ransforms them into display coordinates (pixel) for higher precision of assignment
        distances = []
        for p in pts_all: # iterates through each point
            point_disp = ax.transData.transform((p[0], p[1])) # transforms each point coordinates into display coordinates
            d = np.hypot(click_disp[0] - point_disp[0], click_disp[1] - point_disp[1]) # calculates distance between each point and the click
            distances.append(d) 
        distances = np.array(distances)
        if len(distances) == 0: # if no points exists, return
            return
        i = np.argmin(distances) # gives index of point with smallest distance to the click
        threshold_pixels = 10  # changeable threshold for pixel distance to select point
        if distances[i] < threshold_pixels: # if disnatce between point and click is lower than threshold, select it
            sel = pts_all[i]  # (Distance, Depth, Index, Original color) - inforation about selected point
            if sel[2] in markers: # checks if point is already selected as faulty
                markers[sel[2]].remove()
                del markers[sel[2]]
                if sel[2] in bad_indices: # if already selected as faulty, unselect
                    bad_indices.remove(sel[2])
            else: # if point was not selected before
                marker, = ax.plot(sel[0], sel[1], 'ro', markersize=8) # mark point with red
                markers[sel[2]] = marker # save selection in markers
                bad_indices.append(sel[2]) # add point to bad_indices list
            fig.canvas.draw() # update plot
    
    # RectangleSelector-Callback: selects all points in rectangle
    def onselect(eclick, erelease): # ecklick contains coordinate of click start - erelease coordinates of cklick release
        x_min, x_max = sorted([eclick.xdata, erelease.xdata]) # extract coordinates of click and release
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])
        for p in pts_all: # contains all points of current survey in (distance, depth, index, originalcolor)
            if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max: # checks if point is withign rectangle
                if p[2] in markers: # if index of point is already in markers remove selection and remove from list of bad indices
                    markers[p[2]].remove()
                    del markers[p[2]]
                    if p[2] in bad_indices: 
                        bad_indices.remove(p[2])
                else:
                    marker, = ax.plot(p[0], p[1], 'ro', markersize=8) # if not already selected - turn red and
                    markers[p[2]] = marker
                    bad_indices.append(p[2]) # add to bad indices
        fig.canvas.draw() # update figure
    
    # link and activate cklick and rectangle with mouse click
    fig.canvas.mpl_connect('button_press_event', on_click) # when a mouse click on the plot gets detected on click gets activated
    rect_selector = RectangleSelector(ax, onselect, useblit=True, # initalise rectangle selector in matplot
                                      button=[1],          # left mouse click
                                      minspanx=5, minspany=5,  # min size in pixels
                                      spancoords='pixels',
                                      interactive=True)
    
    plt.show()  # after closing the survey plot, the next one is shown

# After editing, df gets filtered with the indices
df_corrected = df.drop(bad_indices)
# the aritifcal boundary points get added again
df_corrected = df_corrected.append(boundary_points, ignore_index=True)

# save faulty indice points
df.loc[bad_indices].to_csv(FILTER_CSV)
print(f"{len(bad_indices)} points were removed.")