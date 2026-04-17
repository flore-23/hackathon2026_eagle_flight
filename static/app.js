/*  app.js — Eagle Flight Visualizer
 *
 *  Left  : CesiumJS 3D globe, camera locked behind eagle, real terrain.
 *  Right : Leaflet 2D map (default) or Cesium 3D view (toggle).
 *  Bottom: Custom HTML timeline with orange orientation-data markers.
 */

/* ------------------------------------------------------------------ */
/*  Cesium Ion token (optional)                                        */
/*  Paste a free token from https://ion.cesium.com/signup/ for        */
/*  real terrain elevation.  Leave blank for flat terrain + OSM.      */
/* ------------------------------------------------------------------ */
// CESIUM_ION_TOKEN is injected by /config.js (served from .env, never committed)
const CESIUM_ION_TOKEN = window.CESIUM_ION_TOKEN || "";
const HAS_ION = CESIUM_ION_TOKEN.length > 0;
if (HAS_ION) Cesium.Ion.defaultAccessToken = CESIUM_ION_TOKEN;

/* Camera chase parameters */
const CAM_RANGE = 200;    // metres from eagle
const CAM_PITCH = -25;    // degrees (negative = looking down from behind)

/* ------------------------------------------------------------------ */
/*  State                                                              */
/* ------------------------------------------------------------------ */
let flightData      = null;
let viewer3D        = null;
let viewer3DMap     = null;   // lazy second Cesium for "3D map" toggle
let lmap            = null;   // Leaflet instance
let leafletMarker   = null;
let leafletPolylines = [];    // [{polyline, pts}] for click-to-jump
let eagleEntity     = null;
let positionProp    = null;
let orientationProp = null;
let speedSampled    = null;
let headingSampled  = null;
let epochJulian     = null;
let totalSec        = 0;
let tlTracking      = true;   // timeline scrubber follows clock

/* ------------------------------------------------------------------ */
/*  Upload / demo                                                      */
/* ------------------------------------------------------------------ */
const gpsInput    = document.getElementById("gps-file");
const orientInput = document.getElementById("orient-file");
const uploadBtn   = document.getElementById("upload-btn");
const demoBtn     = document.getElementById("demo-btn");
const statusEl    = document.getElementById("upload-status");


gpsInput.addEventListener("change",    () => { uploadBtn.disabled = !(gpsInput.files.length && orientInput.files.length); });
orientInput.addEventListener("change", () => { uploadBtn.disabled = !(gpsInput.files.length && orientInput.files.length); });

uploadBtn.addEventListener("click", async () => {
  setStatus("Merging data – this may take ~10 s…");
  uploadBtn.disabled = true;
  const form = new FormData();
  form.append("gps_file",         gpsInput.files[0]);
  form.append("orientation_file", orientInput.files[0]);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    flightData = await res.json();
    startApp();
  } catch (e) {
    setStatus("Error: " + e.message);
    uploadBtn.disabled = false;
  }
});

demoBtn.addEventListener("click", async () => {
  setStatus("Loading demo data…");
  demoBtn.disabled = true;
  try {
    const res = await fetch("/api/demo");
    if (!res.ok) throw new Error("Demo data not available – run merge_data.py first.");
    flightData = await res.json();
    startApp();
  } catch (e) {
    setStatus("Error: " + e.message);
    demoBtn.disabled = false;
  }
});

function setStatus(msg) { statusEl.textContent = msg; }

/* ------------------------------------------------------------------ */
/*  Helpers: check whether a time (seconds-since-epoch) has 3D data   */
/* ------------------------------------------------------------------ */
function hasOrientationAt(tSec) {
  for (const [s, e] of flightData.orientation_intervals) {
    if (tSec >= s && tSec <= e) return true;
    if (s > tSec) break;
  }
  return false;
}

/* ------------------------------------------------------------------ */
/*  Build CesiumJS sampled properties                                  */
/* ------------------------------------------------------------------ */
function buildPositionProp(positions, epoch) {
  const prop = new Cesium.SampledPositionProperty();
  prop.setInterpolationOptions({
    interpolationDegree: 1,
    interpolationAlgorithm: Cesium.LinearApproximation,
  });
  for (let i = 0; i < positions.length; i += 4) {
    prop.addSample(
      Cesium.JulianDate.addSeconds(epoch, positions[i], new Cesium.JulianDate()),
      Cesium.Cartesian3.fromDegrees(positions[i+1], positions[i+2], positions[i+3])
    );
  }
  return prop;
}

function buildScalarProp(flat, epoch) {
  const prop = new Cesium.SampledProperty(Number);
  prop.setInterpolationOptions({
    interpolationDegree: 1,
    interpolationAlgorithm: Cesium.LinearApproximation,
  });
  for (let i = 0; i < flat.length; i += 2) {
    prop.addSample(
      Cesium.JulianDate.addSeconds(epoch, flat[i], new Cesium.JulianDate()),
      flat[i+1]
    );
  }
  return prop;
}

/*
 * Precompute orientation quaternions for every timestamp.
 *
 * - GPS-only timestamps : heading from GPS, pitch=0, roll=0 (level flight)
 * - Orientation timestamps: GPS heading + IMU pitch + IMU roll
 *
 * Returns a Cesium.SampledProperty<Quaternion>.
 */
function buildOrientationProp(posProp, headProp, epoch) {
  const prop = new Cesium.SampledProperty(Cesium.Quaternion);
  prop.setInterpolationOptions({
    interpolationDegree: 1,
    interpolationAlgorithm: Cesium.LinearApproximation,
  });

  /* Collect ALL keyframes first, then sort by time.
     SampledProperty.addSample() expects chronological order — adding
     GPS-only frames then orientation frames would leave them interleaved
     and break the binary-search interpolation. */
  const keyframes = [];

  // GPS-only keyframes → level flight (pitch=0, roll=0)
  const pos = flightData.positions;
  for (let i = 0; i < pos.length; i += 4) {
    const t = pos[i];
    if (!hasOrientationAt(t)) keyframes.push({ t, pitch: 0, roll: 0 });
  }

  // Orientation keyframes → real body attitude
  const { times, pitch, roll } = flightData.orientations;
  for (let i = 0; i < times.length; i++) {
    keyframes.push({ t: times[i], pitch: pitch[i], roll: roll[i] });
  }

  // Sort chronologically before adding
  keyframes.sort((a, b) => a.t - b.t);

  for (const kf of keyframes) {
    const jt  = Cesium.JulianDate.addSeconds(epoch, kf.t, new Cesium.JulianDate());
    const p   = posProp.getValue(jt);
    if (!p) continue;
    const hDeg = headProp.getValue(jt) || 0;
    const hpr  = new Cesium.HeadingPitchRoll(
      Cesium.Math.toRadians(hDeg),
      Cesium.Math.toRadians(kf.pitch),
      Cesium.Math.toRadians(kf.roll)
    );
    prop.addSample(jt, Cesium.Transforms.headingPitchRollQuaternion(p, hpr));
  }

  return prop;
}

/* ------------------------------------------------------------------ */
/*  Cesium 3D viewer (left pane)                                       */
/* ------------------------------------------------------------------ */
function init3DViewer() {
  const commonOpts = {
    timeline: false, animation: false, baseLayerPicker: false,
    geocoder: false, homeButton: false, sceneModePicker: false,
    navigationHelpButton: false, fullscreenButton: false,
    selectionIndicator: false, infoBox: false, requestRenderMode: false,
  };

  if (HAS_ION) {
    // Modern Cesium 1.107+ API: terrain option + Ion default imagery (Bing satellite)
    viewer3D = new Cesium.Viewer("cesium-3d", {
      ...commonOpts,
      terrain: Cesium.Terrain.fromWorldTerrain(),
    });
  } else {
    viewer3D = new Cesium.Viewer("cesium-3d", {
      ...commonOpts,
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      imageryProvider: new Cesium.UrlTemplateImageryProvider({
        url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        credit: "© OpenStreetMap contributors",
        maximumLevel: 19,
      }),
    });
  }

  viewer3D.scene.globe.depthTestAgainstTerrain = HAS_ION;
  viewer3D.scene.skyAtmosphere.show = true;
}

/* ------------------------------------------------------------------ */
/*  Eagle entity                                                       */
/* ------------------------------------------------------------------ */
function addEagleEntity() {
  eagleEntity = viewer3D.entities.add({
    name: flightData.bird_id,
    position:    positionProp,
    orientation: orientationProp,
    point: {
      pixelSize: 14,
      color: Cesium.Color.ORANGE,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 2,
    },
    path: {
      leadTime:  0,
      trailTime: totalSec,   // full trajectory trail
      width: 2,
      material: new Cesium.PolylineGlowMaterialProperty({
        glowPower: 0.25,
        color: Cesium.Color.ORANGE,
      }),
    },
  });
}

/* ------------------------------------------------------------------ */
/*  Camera: locked behind and above the eagle                          */
/*  Called every Cesium clock tick.                                    */
/* ------------------------------------------------------------------ */
function updateCamera(time) {
  const pos = eagleEntity.position.getValue(time, new Cesium.Cartesian3());
  if (!pos) return;

  const hDeg = headingSampled ? headingSampled.getValue(time) : null;
  if (hDeg == null) return;

  /* HeadingPitchRange works in the local ENU frame at target — no
     reference-frame conversion needed, so the view stays stable.
     heading = eagle's heading → camera is directly behind
     pitch   = negative       → camera looks down from above         */
  viewer3D.camera.lookAt(pos, new Cesium.HeadingPitchRange(
    Cesium.Math.toRadians(hDeg),
    Cesium.Math.toRadians(CAM_PITCH),
    CAM_RANGE
  ));
}

/* ------------------------------------------------------------------ */
/*  Leaflet 2D map (right pane)                                        */
/* ------------------------------------------------------------------ */
function initLeafletMap() {
  lmap = L.map("leaflet-map", { zoomControl: true });

  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(lmap);

  /* Build coloured trajectory segments */
  const positions = flightData.positions;
  const pts = [];
  for (let i = 0; i < positions.length; i += 4) {
    pts.push({ t: positions[i], lat: positions[i+2], lng: positions[i+1] });
  }

  let segStart = 0;
  let curHasOrient = hasOrientationAt(pts[0].t);

  const addLeafletSegment = (from, to, hasOrient) => {
    const latLngs = pts.slice(from, to + 1).map(p => [p.lat, p.lng]);
    if (latLngs.length < 2) return;
    const pl = L.polyline(latLngs, {
      color:   hasOrient ? "#e67e22" : "#888",
      weight:  hasOrient ? 3 : 2,
      opacity: hasOrient ? 0.9 : 0.5,
    }).addTo(lmap);
    /* store points for click-to-jump */
    leafletPolylines.push({ polyline: pl, pts: pts.slice(from, to + 1) });
    pl.on("click", (e) => {
      jumpToNearestPoint(e.latlng.lat, e.latlng.lng);
    });
  };

  for (let i = 1; i < pts.length; i++) {
    const ho = hasOrientationAt(pts[i].t);
    if (ho !== curHasOrient) {
      addLeafletSegment(segStart, i, curHasOrient);
      segStart = i;
      curHasOrient = ho;
    }
  }
  addLeafletSegment(segStart, pts.length - 1, curHasOrient);

  /* Moving position marker */
  leafletMarker = L.circleMarker([pts[0].lat, pts[0].lng], {
    radius: 7, color: "#fff", fillColor: "#e74c3c",
    fillOpacity: 1, weight: 2,
  }).addTo(lmap);

  /* Click anywhere on map background to jump */
  lmap.on("click", (e) => {
    jumpToNearestPoint(e.latlng.lat, e.latlng.lng);
  });

  /* Fit map to trajectory */
  const allLats = pts.map(p => p.lat);
  const allLngs = pts.map(p => p.lng);
  lmap.fitBounds([
    [Math.min(...allLats), Math.min(...allLngs)],
    [Math.max(...allLats), Math.max(...allLngs)],
  ]);
}

function jumpToNearestPoint(lat, lng) {
  /* Find closest trajectory point and jump clock to its timestamp */
  const positions = flightData.positions;
  let bestDist = Infinity, bestT = 0;
  for (let i = 0; i < positions.length; i += 4) {
    const dLat = positions[i+2] - lat;
    const dLng = positions[i+1] - lng;
    const d = dLat * dLat + dLng * dLng;
    if (d < bestDist) { bestDist = d; bestT = positions[i]; }
  }
  const jt = Cesium.JulianDate.addSeconds(epochJulian, bestT, new Cesium.JulianDate());
  viewer3D.clock.currentTime = jt;
  viewer3D.clock.shouldAnimate = false;
  document.getElementById("play-btn").innerHTML = "&#9654;";
}

function updateLeafletMarker(time) {
  if (!lmap || !leafletMarker) return;
  const pos = eagleEntity.position.getValue(time, new Cesium.Cartesian3());
  if (!pos) return;
  const carto = Cesium.Cartographic.fromCartesian(pos);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  const lng = Cesium.Math.toDegrees(carto.longitude);
  leafletMarker.setLatLng([lat, lng]);
}

/* ------------------------------------------------------------------ */
/*  Lazy Cesium 3D map (right pane "3D" toggle)                        */
/* ------------------------------------------------------------------ */
function initCesiumMapViewer() {
  if (viewer3DMap) return;  // already initialised

  const mapOpts = {
    timeline: false, animation: false, baseLayerPicker: false,
    geocoder: false, homeButton: false, sceneModePicker: false,
    navigationHelpButton: false, fullscreenButton: false,
    selectionIndicator: false, infoBox: false, requestRenderMode: false,
  };
  if (HAS_ION) {
    viewer3DMap = new Cesium.Viewer("cesium-map", {
      ...mapOpts, terrain: Cesium.Terrain.fromWorldTerrain(),
    });
  } else {
    viewer3DMap = new Cesium.Viewer("cesium-map", {
      ...mapOpts, terrainProvider: new Cesium.EllipsoidTerrainProvider(),
    });
    viewer3DMap.imageryLayers.removeAll();
    viewer3DMap.imageryLayers.addImageryProvider(
      new Cesium.UrlTemplateImageryProvider({
        url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        credit: "© OpenStreetMap contributors", maximumLevel: 19,
      })
    );
  }

  /* Draw the trajectory as a polyline entity */
  const pos3d = flightData.positions;
  const pts = [];
  for (let i = 0; i < pos3d.length; i += 4) {
    pts.push(Cesium.Cartesian3.fromDegrees(pos3d[i+1], pos3d[i+2], pos3d[i+3]));
  }
  viewer3DMap.entities.add({
    polyline: { positions: pts, width: 3, material: Cesium.Color.ORANGE },
  });

  /* Moving marker synced to clock */
  const mapMarker = viewer3DMap.entities.add({
    position: positionProp,
    point: { pixelSize: 10, color: Cesium.Color.RED },
  });

  /* Sync clock */
  viewer3DMap.clock.startTime   = viewer3D.clock.startTime.clone();
  viewer3DMap.clock.stopTime    = viewer3D.clock.stopTime.clone();
  viewer3DMap.clock.currentTime = viewer3D.clock.currentTime.clone();
  viewer3D.clock.onTick.addEventListener(() => {
    viewer3DMap.clock.currentTime = viewer3D.clock.currentTime.clone();
  });

  /* Click-to-jump using ScreenSpaceEventHandler */
  const handler = new Cesium.ScreenSpaceEventHandler(viewer3DMap.scene.canvas);
  handler.setInputAction((e) => {
    const ray = viewer3DMap.camera.getPickRay(e.position);
    const cart = viewer3DMap.scene.globe.pick(ray, viewer3DMap.scene)
              || viewer3DMap.scene.pickPosition(e.position);
    if (!cart) return;
    const carto = Cesium.Cartographic.fromCartesian(cart);
    jumpToNearestPoint(
      Cesium.Math.toDegrees(carto.latitude),
      Cesium.Math.toDegrees(carto.longitude)
    );
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

  viewer3DMap.zoomTo(viewer3DMap.entities);
}

/* ------------------------------------------------------------------ */
/*  View toggle (2D Leaflet / 3D Cesium map)                           */
/* ------------------------------------------------------------------ */
function setupViewToggle() {
  document.querySelectorAll(".view-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-toggle").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      if (btn.dataset.mode === "2d") {
        document.getElementById("leaflet-map").classList.remove("hidden");
        document.getElementById("cesium-map").classList.add("hidden");
        lmap.invalidateSize();
      } else {
        document.getElementById("leaflet-map").classList.add("hidden");
        document.getElementById("cesium-map").classList.remove("hidden");
        initCesiumMapViewer();   // lazy-init
      }
    });
  });
}

/* ------------------------------------------------------------------ */
/*  Custom HTML timeline                                               */
/* ------------------------------------------------------------------ */
function buildTimeline() {
  const track    = document.getElementById("tl-track");
  const scrubber = document.getElementById("tl-scrubber");
  const tCurrent = document.getElementById("tl-time-current");
  const tEnd     = document.getElementById("tl-time-end");

  const start  = viewer3D.clock.startTime;
  const stop   = viewer3D.clock.stopTime;
  totalSec     = Cesium.JulianDate.secondsDifference(stop, start);

  /* End time label */
  tEnd.textContent = formatDuration(totalSec);

  /* Orange interval markers */
  for (const [s, e] of flightData.orientation_intervals) {
    const div = document.createElement("div");
    div.className = "tl-interval";
    div.style.left  = (s / totalSec * 100).toFixed(3) + "%";
    div.style.width = ((e - s) / totalSec * 100).toFixed(3) + "%";
    track.insertBefore(div, scrubber);
  }

  /* Clock tick → move scrubber */
  viewer3D.clock.onTick.addEventListener((clock) => {
    const elapsed = Cesium.JulianDate.secondsDifference(clock.currentTime, start);
    const frac    = Math.max(0, Math.min(1, elapsed / totalSec));
    scrubber.style.left = (frac * 100).toFixed(3) + "%";
    tCurrent.textContent = formatDuration(elapsed);

    /* HUD */
    const spd = speedSampled ? speedSampled.getValue(clock.currentTime) : null;
    const pos = eagleEntity ? eagleEntity.position.getValue(clock.currentTime) : null;
    let alt = null;
    if (pos) {
      alt = Cesium.Cartographic.fromCartesian(pos).height;
    }
    document.getElementById("hud-speed").textContent =
      spd != null ? `Speed: ${spd.toFixed(1)} m/s` : "Speed: —";
    document.getElementById("hud-alt").textContent =
      alt != null ? `Altitude: ${alt.toFixed(0)} m`  : "Altitude: —";

    /* Camera + Leaflet marker */
    updateCamera(clock.currentTime);
    updateLeafletMarker(clock.currentTime);
  });

  /* Click on track → seek */
  const seek = (clientX) => {
    const rect = track.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const jt   = Cesium.JulianDate.addSeconds(start, frac * totalSec, new Cesium.JulianDate());
    viewer3D.clock.currentTime = jt;
  };

  track.addEventListener("mousedown", (e) => {
    seek(e.clientX);
    const onMove = (ev) => seek(ev.clientX);
    const onUp   = ()   => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup",   onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup",   onUp);
  });
}

function formatDuration(sec) {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(ss).padStart(2,"0")}`;
}

/* ------------------------------------------------------------------ */
/*  Play / pause + speed                                               */
/* ------------------------------------------------------------------ */
function setupPlayback() {
  const playBtn  = document.getElementById("play-btn");
  const speedSel = document.getElementById("speed-select");

  playBtn.addEventListener("click", () => {
    viewer3D.clock.shouldAnimate = !viewer3D.clock.shouldAnimate;
    playBtn.innerHTML = viewer3D.clock.shouldAnimate ? "&#9646;&#9646;" : "&#9654;";
  });

  speedSel.addEventListener("change", () => {
    viewer3D.clock.multiplier = Number(speedSel.value);
  });
}

/* ------------------------------------------------------------------ */
/*  Main startup                                                       */
/* ------------------------------------------------------------------ */
async function startApp() {
  setStatus("Building visualisation…");

  /* Show app, hide overlay */
  document.getElementById("upload-overlay").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");

  epochJulian = Cesium.JulianDate.fromIso8601(flightData.epoch);
  const startJD = Cesium.JulianDate.fromIso8601(flightData.start_time);
  const stopJD  = Cesium.JulianDate.fromIso8601(flightData.end_time);

  /* Build sampled properties */
  setStatus("Building position data…");
  positionProp   = buildPositionProp(flightData.positions, epochJulian);
  speedSampled   = buildScalarProp(flightData.speeds,   epochJulian);
  headingSampled = buildScalarProp(flightData.headings, epochJulian);

  setStatus("Precomputing orientation quaternions (this takes a few seconds)…");
  /* Yield to browser so "Building…" message renders before the heavy loop */
  await new Promise(r => setTimeout(r, 30));
  orientationProp = buildOrientationProp(positionProp, headingSampled, epochJulian);

  /* Cesium viewer */
  setStatus("Initialising 3D viewer…");
  await new Promise(r => setTimeout(r, 30));
  init3DViewer();

  /* Clock */
  viewer3D.clock.startTime    = startJD.clone();
  viewer3D.clock.stopTime     = stopJD.clone();
  viewer3D.clock.currentTime  = startJD.clone();
  viewer3D.clock.clockRange   = Cesium.ClockRange.LOOP_STOP;
  viewer3D.clock.multiplier   = 1;
  viewer3D.clock.shouldAnimate = false;

  /* Compute total duration (needed for trail length) */
  totalSec = Cesium.JulianDate.secondsDifference(stopJD, startJD);

  /* Eagle */
  addEagleEntity();

  /* Leaflet 2D map */
  initLeafletMap();

  /* Timeline + playback */
  buildTimeline();
  setupPlayback();
  setupViewToggle();

  /* Start playing */
  viewer3D.clock.shouldAnimate = true;
  document.getElementById("play-btn").innerHTML = "&#9646;&#9646;";

  setStatus("");
}
