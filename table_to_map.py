import numpy as np
import folium
import pandas as pd
from geopy.geocoders import Nominatim
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon



def map_gen(df1):
    mapObj = folium.Map(location=[35.0211, 135.754],zoom_start=5,width=1300, height=800)
    for index, row in df1.iterrows():
        # print(row["Date"])
        folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=5,
                            popup=folium.Popup(str(row["Date"]) + "\n" + str(row["Time"]) + "\n" + str(row["Summary"])
                                               , parse_html=True, max_width=200),
                            fill_color="red",
                            color="gray",
                            fill_opacity=0.7).add_to(mapObj)
    return mapObj