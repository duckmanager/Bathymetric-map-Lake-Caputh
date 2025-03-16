import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gps_boxplots(used_gps_points):

    stats = pd.DataFrame({
    'mean DOP lon': used_gps_points['DOP (lon)'].mean(),
    'mean DOP lat': used_gps_points['DOP (lat)'].mean(),
    'mean VDOP': used_gps_points['VDOP'].mean(),

    'STD DOP lon': used_gps_points['DOP (lon)'].std(),
    'STD DOP lat': used_gps_points['DOP (lat)'].std(),
    'STD VDOP': used_gps_points['VDOP'].std(),
    }, index=[""])

    print(stats)





    plt.figure(figsize=(10, 6))
    
    # Boxplots erstellen
    plt.boxplot([used_gps_points['DOP (lon)'], used_gps_points['DOP (lat)'], used_gps_points['VDOP']], 
                labels=["DOP (lon)", "DOP (lat)", "VDOP"])
    
    # Titel und Achsenbeschriftungen setzen
    plt.title("Verteilung von DOP und VDOP Werten")
    plt.ylabel("Standartabweichung einzelner Punkte in m")
    
    # Anzahl der Punkte oben rechts einfügen
    num_points = len(used_gps_points)
    max_value = max(used_gps_points[['DOP (lon)', 'DOP (lat)', 'VDOP']].max())
    plt.text(2.5, max_value, f"Points: {num_points}", 
             ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Anzeigen
    plt.show()




    filtered_gps_points = used_gps_points[used_gps_points['VDOP'] <= 0.02]

    # Gruppieren nach Datum und Berechnungen durchführen
    aggregated_data = filtered_gps_points.groupby('date').agg(
        avg_hgt=('hgt', 'mean'),       # Durchschnittliche Höhe
        count_points=('hgt', 'count'), # Anzahl der Punkte
        avg_vdop=('VDOP', 'mean')      # Durchschnittlicher VDOP
    ).reset_index()

    # Ausgabe des Ergebnisses
    print(aggregated_data)







def main():
    data_dir = Path("output/multibeam/QC")
    print("reading data")
    used_gps_points = pd.read_csv(data_dir/"used_GPS_points.csv")

    print("plot DOP and VDOP")
    plot_gps_boxplots(used_gps_points)

    input("done!")

main()