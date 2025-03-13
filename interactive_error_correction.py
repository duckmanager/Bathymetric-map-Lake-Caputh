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

bad_indices = []  # collection of faulty indices 

# Iterate through all surveys showing progress with tqdm
for fid in tqdm(df['file_id'].unique(), desc="Messfahrten"):
    subdf = df[df['file_id'] == fid]
    pts_all = []  # Liste aller Punkte: (Distanz, Tiefe, Index, ursprüngliche Farbe)
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {}  # Speichert Marker für das Toggle-Verhalten
    beams = subdf['Beam_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(beams)))
    
    for color, beam in zip(colors, beams):
        beam_df = subdf[subdf['Beam_type'] == beam].sort_index()
        if beam_df.empty:
            continue
        x_coords = beam_df['Longitude'].values
        y_coords = beam_df['Latitude'].values
        # Berechnung der kumulativen Distanz (in Metern)
        dist = np.r_[0, np.cumsum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))]
        depth = beam_df['Depth (m)'].values
        ax.scatter(dist, depth, label=beam, color=color)
        for d, dep, idx in zip(dist, depth, beam_df.index):
            pts_all.append((d, dep, idx, color))
    
    ax.set_xlabel("Fahrstrecke (m)")
    ax.set_ylabel("Tiefe (m)")
    if pts_all:
        min_depth = min(p[1] for p in pts_all)
        # Y-Achse: oberster Wert 0, unterster Wert = minimaler Tiefenwert minus 0.5 m Puffer
        ax.set_ylim(min_depth - 0.5, 0)
    ax.legend()
    plt.title(f"Messfahrt: {fid}")
    
    # Klick-Event: wählt den nächstgelegenen Punkt (in Displaykoordinaten) aus
    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        click_disp = ax.transData.transform((event.xdata, event.ydata))
        distances = []
        for p in pts_all:
            point_disp = ax.transData.transform((p[0], p[1]))
            d = np.hypot(click_disp[0] - point_disp[0], click_disp[1] - point_disp[1])
            distances.append(d)
        distances = np.array(distances)
        if len(distances) == 0:
            return
        i = np.argmin(distances)
        threshold_pixels = 10  # Anpassbarer Schwellwert in Pixeln
        if distances[i] < threshold_pixels:
            sel = pts_all[i]  # (Distanz, Tiefe, Index, Originalfarbe)
            if sel[2] in markers:
                markers[sel[2]].remove()
                del markers[sel[2]]
                if sel[2] in bad_indices:
                    bad_indices.remove(sel[2])
            else:
                marker, = ax.plot(sel[0], sel[1], 'ro', markersize=8)
                markers[sel[2]] = marker
                bad_indices.append(sel[2])
            fig.canvas.draw()
    
    # RectangleSelector-Callback: toggelt alle Punkte im gezogenen Rechteck
    def onselect(eclick, erelease):
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])
        for p in pts_all:
            if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                if p[2] in markers:
                    markers[p[2]].remove()
                    del markers[p[2]]
                    if p[2] in bad_indices:
                        bad_indices.remove(p[2])
                else:
                    marker, = ax.plot(p[0], p[1], 'ro', markersize=8)
                    markers[p[2]] = marker
                    bad_indices.append(p[2])
        fig.canvas.draw()
    
    # Beide Event-Handler aktivieren
    fig.canvas.mpl_connect('button_press_event', on_click)
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1],          # linke Maustaste
                                      minspanx=5, minspany=5,  # Mindestgröße in Pixeln
                                      spancoords='pixels',
                                      interactive=True)
    
    plt.show()  # Nach Schließen des Fensters wird der Plot der nächsten Messfahrt angezeigt

# Nach der Bearbeitung: Filtere die fehlerhaften Punkte und speichere sie als CSV
df_corrected = df.drop(bad_indices)
# Füge die artificial_boundary_points wieder hinzu
df_corrected = df_corrected.append(boundary_points, ignore_index=True)

df.loc[bad_indices].to_csv(FILTER_CSV)
print(f"{len(bad_indices)} Punkte wurden gefiltert. Die korrigierten Daten können weiterverarbeitet werden.")