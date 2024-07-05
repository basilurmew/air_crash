import numpy as np
import folium
import pandas as pd
from geopy.geocoders import Nominatim
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon



ressons_dict = {"weather":0, "engine":1, "people":2, "others":3}

color_dict = {"weather":"yellow", "engine":"red", "people":"blue", "others":"green"}



def data_condition(df1, lower_year, high_year):
    dct = {"Date":[],"Time":[], "Location":[] ,"latitude":[], "longitude":[],"Summary":[],"kmeans":[]}
    res = pd.DataFrame(dct)
    for index, row in df1.iterrows():
        if int(row["Date"][-4:]) > high_year:
            break
        if int(row["Date"][-4:]) >= lower_year:
            l = [row["Date"], row["Time"], row["Location"], row["latitude"],row["longitude"], row["Summary"],row["kmeans"]]
            res.loc[len(res.index )] = l
    return res

#def reason condition():


def map_gen(df1):
    mapObj = folium.Map(location=[35.0211, 135.754],zoom_start=5,width=1300, height=800)
    for index, row in df1.iterrows():
        folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=5,
                            popup=folium.Popup(str(row["Date"]) + "\n" + str(row["Time"]) + "\n" + str(row["Summary"])
                                               , parse_html=True, max_width=200),
                            fill_color="red",
                            color="gray",
                            fill_opacity=0.7).add_to(mapObj)
    return mapObj