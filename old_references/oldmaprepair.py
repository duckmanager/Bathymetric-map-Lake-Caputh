import geopandas as gpd
from shapely.validation import make_valid
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Polygon, MultiPolygon

# Pfad zur ursprünglichen Shapefile
shapefile_path = "old_map/1/capsee_repaired.shp"

# Shapefile laden
gdf = gpd.read_file(shapefile_path)
gdf.to_crs("EPSG:25833", inplace=True)


# Funktion zur Korrektur der X-Koordinaten
def correct_geometry(geom):
    if geom.geom_type == "Polygon":
        corrected_exterior = [(float(str(x)[1:]), y) for x, y in geom.exterior.coords]
        corrected_interiors = [[(float(str(x)[1:]), y) for x, y in interior.coords] for interior in geom.interiors]
        return Polygon(corrected_exterior, corrected_interiors)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([correct_geometry(poly) for poly in geom.geoms])
    return geom  # Falls es sich um einen anderen Geometrietyp handelt, bleibt er unverändert

# Korrektur anwenden
gdf["geometry"] = gdf["geometry"].apply(correct_geometry)

# Shapefile mit korrigierten Geometrien speichern
gdf.to_file(shapefile_path)
input()






# Geometrie reparieren
# Geometrien extrahieren und einer eigenen Variable zuweisen
geometries = gdf["geometry"]

import geopandas as gpd
import folium
from shapely.geometry import mapping

# 📌 Shapefile laden

# 🗺 Falls kein CRS definiert ist, WGS84 setzen (OSM benötigt EPSG:4326)
if gdf.crs is None:
    gdf.set_crs("EPSG:3857", inplace=True)
elif gdf.crs.to_string() != "EPSG:3857":
    gdf = gdf.to_crs(epsg=4326)  # Falls nötig, umprojizieren

# 🔎 Mittelpunkt der Karte berechnen (Mittelwert der Bounding Box)
bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

# 🗺 Interaktive Karte mit Folium erstellen
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

# 📌 Shapefile als Layer hinzufügen
folium.GeoJson(
    gdf, 
    name="Shapefile-Layer",
    style_function=lambda x: {
        "color": "blue", 
        "weight": 2, 
        "fillOpacity": 0.1
    }
).add_to(m)

# 🖱 Interaktive Layer-Steuerung hinzufügen
folium.LayerControl().add_to(m)




m.save("map_interactive.html")

print(gdf.crs)

# Speichere die reparierte Shapefile
repaired_shapefile_path = "capsee_repaired_fixed.shp"
gdf.to_file(repaired_shapefile_path)

print(f"Reparierte Datei gespeichert unter: {repaired_shapefile_path}")
