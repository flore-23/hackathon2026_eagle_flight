import streamlit as st
import pandas as pd
import json
import base64
import streamlit.components.v1 as components

st.set_page_config(page_title="Eagle Flight", layout="wide")


def load_data():
    df = pd.read_csv("Data/madi_loc_day1.csv", sep=";")
    df.columns = df.columns.str.strip()
    df["lat"] = df["lat"].astype(str).str.replace(",", ".", regex=False).astype(float)
    df["long"] = df["long"].astype(str).str.replace(",", ".", regex=False).astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").dropna(subset=["lat", "long"]).reset_index(drop=True)
    return df


def png_to_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


df = load_data()
coords = df[["lat", "long"]].values.tolist()
coords_json = json.dumps(coords)
times_json = json.dumps(df["timestamp"].astype(str).tolist())
eagle_icon_url = png_to_data_url("eagle.png")


left, right = st.columns([1, 2])

with left:
    st.title("Eagle Flight")
    st.write("Use the buttons under the map to play, pause, or reset.")
    st.write("Use the layer control on the map to switch between Satellite and Topographic.")
    st.write(f"Total points: {len(df)}")
    st.write(f"Start: {df.iloc[0]['timestamp']}")
    st.write(f"End: {df.iloc[-1]['timestamp']}")

with right:
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
            }}

            #map {{
                width: 100%;
                height: 650px;
            }}

            .leaflet-control.custom-controls {{
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.3);
            }}

            .custom-controls button {{
                margin: 4px 4px 4px 0;
                padding: 6px 10px;
                border: 1px solid #ccc;
                background: white;
                cursor: pointer;
                border-radius: 6px;
            }}

            .custom-controls button:hover {{
                background: #f2f2f2;
            }}

            .custom-controls input[type="range"] {{
                width: 220px;
                vertical-align: middle;
            }}

            .info-text {{
                font-size: 13px;
                margin-top: 6px;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>

        <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
        <script>
            const coords = {coords_json};
            const timestamps = {times_json};
            const eagleIconUrl = "{eagle_icon_url}";

            let index = 0;
            let interval = null;

            const map = L.map("map").setView(coords[0], 10);

            const satellite = L.tileLayer(
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
                {{
                    attribution: "Tiles © Esri"
                }}
            );

            const topo = L.tileLayer(
                "https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png",
                {{
                    attribution: "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap"
                }}
            );

            satellite.addTo(map);

            const baseMaps = {{
                "Satellite": satellite,
                "Topographic": topo
            }};

            L.control.layers(baseMaps, null, {{ collapsed: false }}).addTo(map);

            const eagleIcon = L.icon({{
                iconUrl: eagleIconUrl,
                iconSize: [42, 42],
                iconAnchor: [21, 21]
            }});

            const line = L.polyline([coords[0]], {{
                color: "blue",
                weight: 4,
                opacity: 0.9
            }}).addTo(map);

            const marker = L.marker(coords[0], {{
                icon: eagleIcon
            }}).addTo(map);

            const info = L.control({{ position: "bottomleft" }});
            info.onAdd = function(map) {{
                const div = L.DomUtil.create("div", "leaflet-control custom-controls");
                div.innerHTML = `
                    <div>
                        <button id="playBtn">Play</button>
                        <button id="pauseBtn">Pause</button>
                        <button id="resetBtn">Reset</button>
                    </div>
                    <div style="margin-top:8px;">
                        <input id="progressSlider" type="range" min="0" max="${{coords.length - 1}}" value="0" />
                    </div>
                    <div class="info-text" id="timeLabel">Point 1 / ${{coords.length}}<br>${{timestamps[0]}}</div>
                `;
                return div;
            }};
            info.addTo(map);

            function updateVisuals() {{
                marker.setLatLng(coords[index]);
                line.setLatLngs(coords.slice(0, index + 1));
                document.getElementById("progressSlider").value = index;
                document.getElementById("timeLabel").innerHTML =
                    `Point ${{index + 1}} / ${{coords.length}}<br>${{timestamps[index]}}`;
            }}

            function step() {{
                if (index >= coords.length - 1) {{
                    pause();
                    return;
                }}
                index += 1;
                updateVisuals();
            }}

            function play() {{
                if (interval) return;
                interval = setInterval(step, 60);
            }}

            function pause() {{
                if (interval) {{
                    clearInterval(interval);
                    interval = null;
                }}
            }}

            function resetAnim() {{
                pause();
                index = 0;
                updateVisuals();
                map.panTo(coords[0]);
            }}

            document.addEventListener("click", function(e) {{
                if (e.target && e.target.id === "playBtn") play();
                if (e.target && e.target.id === "pauseBtn") pause();
                if (e.target && e.target.id === "resetBtn") resetAnim();
            }});

            document.addEventListener("input", function(e) {{
                if (e.target && e.target.id === "progressSlider") {{
                    pause();
                    index = parseInt(e.target.value);
                    updateVisuals();
                }}
            }});

            updateVisuals();
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=680, scrolling=False)