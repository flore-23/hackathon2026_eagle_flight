import streamlit as st
from streamlit_folium import st_folium 
import xyzservices.providers as xyz
import folium
st.set_page_config(page_title="Eagle Flight", layout="wide")
m = folium.Map(location=[40, 10], zoom_start=5, tiles=None)

folium.TileLayer(
    tiles=xyz.Esri.WorldImagery,
    name="Satellite",
).add_to(m)

folium.TileLayer(
    tiles=xyz.OpenTopoMap,
    name="Topographic",
).add_to(m)

folium.TileLayer(
    tiles=xyz.CartoDB.Positron,
    name="Light",
).add_to(m)

folium.LayerControl().add_to(m)
left, right = st.columns([2, 1])
with left:
    st.title("Eagle Flight")

with right:
    st_folium(m, width=800, height=600)
