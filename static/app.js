/*  app.js — Eagle Flight Visualizer
 *
 *  Left  : CesiumJS 3D globe, delayed chase camera behind eagle, real terrain.
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
const CAM_RANGE = 90;             // metres from eagle (closer = eagle fills more of screen)
const CAM_PITCH = -20;            // degrees (negative = looking down from behind)
const CAM_HEADING_LAG_SEC = 60;   // view turns one minute after the eagle turns
const CAM_FOCUS_BEHIND_M = 25;    // focus slightly behind the eagle so turns are visible

/*
 * MODEL_HEADING_CORRECTION: constant offset (degrees) added to GPS heading
 * before orienting the 3D model.  The Sketchfab eagle's visible beak is
 * 270° counter-clockwise from Cesium's HPR forward axis after the glTF axis
 * conversion, so adding 270° aligns the beak with the travel direction.
 */
const MODEL_HEADING_CORRECTION = 270;

/*
 * The glTF scene origin is well away from the visible bird.  This extra
 * transform is applied after Eagle(0)'s baked transform and places the tail
 * at the entity position, so Cesium's path emerges from the eagle instead
 * of floating beside it.
 */
const EAGLE_MODEL_ANCHOR_NODE = "Eagle(0)";
const EAGLE_MODEL_ANCHOR_TRANSLATION = new Cesium.Cartesian3(
  99.999917,
  90.688003,
  -75.709188
);

/* ------------------------------------------------------------------ */
/*  Geo helpers: bearing and speed from consecutive GPS positions       */
/* ------------------------------------------------------------------ */

/** Great-circle bearing from point 1 to point 2, degrees [0, 360). */
function geoBearing(lat1, lon1, lat2, lon2) {
  const R2D = 180 / Math.PI, D2R = Math.PI / 180;
  const φ1 = lat1 * D2R, φ2 = lat2 * D2R, Δλ = (lon2 - lon1) * D2R;
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
  return (Math.atan2(y, x) * R2D + 360) % 360;
}

/** Haversine distance in metres between two lat/lon points. */
function geoDistM(lat1, lon1, lat2, lon2) {
  const D2R = Math.PI / 180, R = 6371000;
  const Δφ = (lat2 - lat1) * D2R, Δλ = (lon2 - lon1) * D2R;
  const a = Math.sin(Δφ / 2) ** 2
          + Math.cos(lat1 * D2R) * Math.cos(lat2 * D2R) * Math.sin(Δλ / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function unwrapHeadingDegrees(previous, next) {
  if (previous == null) return next;
  let unwrapped = next;
  while (unwrapped - previous > 180) unwrapped -= 360;
  while (unwrapped - previous < -180) unwrapped += 360;
  return unwrapped;
}

function offsetCartesianMeters(position, headingDeg, forwardM, upM = 0) {
  const headingRad = Cesium.Math.toRadians(headingDeg);
  const eastM = Math.sin(headingRad) * forwardM;
  const northM = Math.cos(headingRad) * forwardM;
  const enu = Cesium.Transforms.eastNorthUpToFixedFrame(position);
  const offset = Cesium.Matrix4.multiplyByPointAsVector(
    enu,
    new Cesium.Cartesian3(eastM, northM, upM),
    new Cesium.Cartesian3()
  );
  return Cesium.Cartesian3.add(position, offset, new Cesium.Cartesian3());
}

/**
 * Build a SampledProperty<Number> for HEADING (degrees) derived purely from
 * consecutive GPS positions.  This is reliable even when the GPS heading
 * column is zero or stale (e.g. 15-minute fix intervals).
 */
function buildHeadingFromPositions(positions, epoch) {
  const prop = new Cesium.SampledProperty(Number);
  prop.setInterpolationOptions({
    interpolationDegree: 1,
    interpolationAlgorithm: Cesium.LinearApproximation,
  });
  const n = positions.length;
  let previous = null;
  for (let i = 0; i < n; i += 4) {
    const t   = positions[i];
    const lon = positions[i + 1], lat = positions[i + 2];
    let bearing;
    if (i + 4 < n) {
      bearing = geoBearing(lat, lon, positions[i + 6], positions[i + 5]);
    } else if (i > 0) {
      // last point: use reverse of segment from previous to this
      bearing = geoBearing(positions[i - 2], positions[i - 3], lat, lon);
    } else {
      bearing = 0;
    }
    bearing = unwrapHeadingDegrees(previous, bearing);
    previous = bearing;
    prop.addSample(
      Cesium.JulianDate.addSeconds(epoch, t, new Cesium.JulianDate()),
      bearing
    );
  }
  return prop;
}

/**
 * Build a SampledProperty<Number> for SPEED (m/s) derived from consecutive
 * GPS positions and the time delta between them.  This correctly handles
 * sparse GPS fixes (e.g. 15 minutes apart) where the recorded speed column
 * may be 0 or stale.
 */
function buildSpeedFromPositions(positions, epoch) {
  const prop = new Cesium.SampledProperty(Number);
  prop.setInterpolationOptions({
    interpolationDegree: 1,
    interpolationAlgorithm: Cesium.LinearApproximation,
  });
  const n = positions.length;
  for (let i = 0; i < n; i += 4) {
    const t   = positions[i];
    const lon = positions[i + 1], lat = positions[i + 2];
    let speed;
    if (i + 4 < n) {
      const dt = positions[i + 4] - t;
      speed = dt > 0 ? geoDistM(lat, lon, positions[i + 6], positions[i + 5]) / dt : 0;
    } else if (i > 0) {
      // last point: copy speed from previous segment
      const dt = t - positions[i - 4];
      speed = dt > 0 ? geoDistM(positions[i - 2], positions[i - 3], lat, lon) / dt : 0;
    } else {
      speed = 0;
    }
    prop.addSample(
      Cesium.JulianDate.addSeconds(epoch, t, new Cesium.JulianDate()),
      speed
    );
  }
  return prop;
}

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
let leafletPastPath = null;
let leafletAllPts = [];

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

  /* ---- Physics-based roll for GPS-only keyframes ----------------------
   *
   * An eagle in a coordinated turn banks at:
   *   bank = atan(v * ω / g)
   * where ω = dHeading/dt (rad/s), v = airspeed (m/s), g = 9.81.
   *
   * Positive ω = clockwise (right turn) → positive roll = right wing down.
   * This produces realistic banking in ALL sections, not just IMU sections.
   * ------------------------------------------------------------------- */
  const G = 9.81;
  const MAX_BANK_DEG = 45;   // cap to avoid extreme values from GPS noise

  // Sample heading and speed at every GPS timestamp
  const pos = flightData.positions;
  const gpsTimes = [], gpsHead = [], gpsSpd = [];
  for (let i = 0; i < pos.length; i += 4) {
    const t  = pos[i];
    const jt = Cesium.JulianDate.addSeconds(epoch, t, new Cesium.JulianDate());
    gpsTimes.push(t);
    gpsHead.push(headProp.getValue(jt) ?? 0);
    gpsSpd.push(speedSampled ? (speedSampled.getValue(jt) ?? 0) : 0);
  }

  // Central-difference heading rate → bank angle at each GPS sample
  const gpsRoll = gpsTimes.map((t, i) => {
    if (i === 0 || i === gpsTimes.length - 1) return 0;
    const dt = gpsTimes[i + 1] - gpsTimes[i - 1];
    if (dt <= 0) return 0;
    let dH = gpsHead[i + 1] - gpsHead[i - 1];
    if (dH >  180) dH -= 360;   // wrap [-180, 180]
    if (dH < -180) dH += 360;
    const omega   = Cesium.Math.toRadians(dH) / dt;      // rad/s
    const bankRad = Math.atan2(gpsSpd[i] * omega, G);    // physics
    return Math.max(-MAX_BANK_DEG, Math.min(MAX_BANK_DEG,
                    Cesium.Math.toDegrees(bankRad)));
  });

  /* ---- Collect ALL keyframes, sort chronologically, then add ----------
   *
   * SampledProperty.addSample() expects chronological order.
   * Building GPS-only then IMU frames separately would interleave them
   * and break the binary-search interpolation.
   * ------------------------------------------------------------------- */
  const keyframes = [];

  // GPS-only keyframes → physics roll, level pitch
  for (let i = 0; i < gpsTimes.length; i++) {
    const t = gpsTimes[i];
    if (!hasOrientationAt(t)) keyframes.push({ t, pitch: 0, roll: gpsRoll[i] });
  }

  // IMU keyframes → real body attitude from sensor
  const { times, pitch, roll } = flightData.orientations;
  for (let i = 0; i < times.length; i++) {
    keyframes.push({ t: times[i], pitch: pitch[i], roll: roll[i] });
  }

  keyframes.sort((a, b) => a.t - b.t);

  for (const kf of keyframes) {
    const jt  = Cesium.JulianDate.addSeconds(epoch, kf.t, new Cesium.JulianDate());
    const p   = posProp.getValue(jt);
    if (!p) continue;
    const hDeg = headProp.getValue(jt) ?? 0;
    const hpr  = new Cesium.HeadingPitchRoll(
      Cesium.Math.toRadians(hDeg + MODEL_HEADING_CORRECTION),
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
/*
 * Scale factor for the 3D model.
 * The FBX was authored in centimetres (bounding box ~191 × 43 × 89 cm),
 * so at glTF's native metre units the model would be 191 m wide.
 * MODEL_SCALE = 0.05 → ~10 m wingspan — clearly visible from 200 m chase cam.
 * Raise toward 0.01 for realistic ~2 m eagle size.
 */
const MODEL_SCALE = 0.1;

function addEagleEntity() {
  eagleEntity = viewer3D.entities.add({
    name: flightData.bird_id,
    position:    positionProp,
    orientation: orientationProp,
    model: {
      uri:            "/models/animated_eagle/scene.gltf",
      scale:          MODEL_SCALE,
      minimumPixelSize: 32,       // never shrinks below 32 px when far away
      maximumScale:   2000,       // cap to avoid huge size near camera
      runAnimations:  false,      // static glide pose — no flapping yet
      nodeTransformations: {
        [EAGLE_MODEL_ANCHOR_NODE]: {
          translation: EAGLE_MODEL_ANCHOR_TRANSLATION,
        },
      },
    },
    path: {
      leadTime:  0,
      trailTime: totalSec,
      resolution: 1,
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

  const currentHeading = headingSampled ? headingSampled.getValue(time) : null;
  if (currentHeading == null) return;

  const elapsed = Cesium.JulianDate.secondsDifference(time, viewer3D.clock.startTime);
  const lagSec = Math.min(CAM_HEADING_LAG_SEC, Math.max(0, elapsed));
  const laggedTime = Cesium.JulianDate.addSeconds(
    time,
    -lagSec,
    new Cesium.JulianDate()
  );
  const hDeg = headingSampled.getValue(laggedTime) ?? currentHeading;
  if (hDeg == null) return;

  const target = offsetCartesianMeters(pos, currentHeading, -CAM_FOCUS_BEHIND_M);

  /* HeadingPitchRange works in the local ENU frame at target — no
     reference-frame conversion needed, so the view stays stable.
     heading = lagged eagle heading → camera follows turns with delay
     pitch   = negative       → camera looks down from above         */
  viewer3D.camera.lookAt(target, new Cesium.HeadingPitchRange(
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

  const positions = flightData.positions;
  const pts = [];

  for (let i = 0; i < positions.length; i += 4) {
    pts.push({
      t: positions[i],
      lat: positions[i + 2],
      lng: positions[i + 1],
    });
  }

  leafletAllPts = pts;

  /* All route: */
  const fullLatLngs = pts.map(p => [p.lat, p.lng]);
  L.polyline(fullLatLngs, {
    color: "#fc9c9c",
    weight: 2,
    opacity: 0.6,
  }).addTo(lmap);

  /* Passed route: */
  leafletPastPath = L.polyline([[pts[0].lat, pts[0].lng]], {
    color: "#ef3737",
    weight: 2,
    opacity: 0.6,
  }).addTo(lmap);

  /* Moving marker */
  leafletMarker = L.circleMarker([pts[0].lat, pts[0].lng], {
    radius: 7,
    color: "#fff",
    fillColor: "#e74c3c",
    fillOpacity: 1,
    weight: 2,
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
  let bestDist = Infinity;
  let bestT = 0;

  for (let i = 0; i < positions.length; i += 4) {
    const dLat = positions[i + 2] - lat;
    const dLng = positions[i + 1] - lng;
    const d = dLat * dLat + dLng * dLng;

    if (d < bestDist) {
      bestDist = d;
      bestT = positions[i];
    }
  }

  const jt = Cesium.JulianDate.addSeconds(
    epochJulian,
    bestT,
    new Cesium.JulianDate()
  );

  viewer3D.clock.currentTime = jt;
  viewer3D.clock.shouldAnimate = false;
  document.getElementById("play-btn").innerHTML = "&#9654;";
}

function updateLeafletMarker(time) {
  if (!lmap || !leafletMarker || !leafletPastPath || !leafletAllPts.length) return;

  const pos = eagleEntity.position.getValue(time, new Cesium.Cartesian3());
  if (!pos) return;

  const carto = Cesium.Cartographic.fromCartesian(pos);
  const lat = Cesium.Math.toDegrees(carto.latitude);
  const lng = Cesium.Math.toDegrees(carto.longitude);

  leafletMarker.setLatLng([lat, lng]);

  const currentT = Cesium.JulianDate.secondsDifference(time, epochJulian);

  const pastPts = leafletAllPts.filter(p => p.t <= currentT);

  const latLngs = pastPts.map(p => [p.lat, p.lng]);
  latLngs.push([lat, lng]);

  leafletPastPath.setLatLngs(latLngs);
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
/*  Plot windows (PlotJuggler-style)                                    */
/* ------------------------------------------------------------------ */
// Variables sharing a unit live in the same color family but get distinct hues
// so they remain distinguishable when overlaid in one plot window.
const PLOT_SERIES = {
  // m — blue family (only one for now)
  altitude:     { label: "altitude (m)",       unit: "m",   color: "#3498db" },
  // m/s — green family (only one for now)
  ground_speed: { label: "ground_speed (m/s)", unit: "m/s", color: "#2ecc71" },
  // deg — warm family, each distinguishable
  roll:         { label: "roll (deg)",         unit: "deg", color: "#e74c3c" }, // red
  pitch:        { label: "pitch (deg)",        unit: "deg", color: "#f1c40f" }, // yellow
  yaw:          { label: "yaw (deg)",          unit: "deg", color: "#e67e22" }, // orange
};

const ORIENT_GAP_THRESH = 5;   // sec — bigger gap → break the line
const seriesDataCache = {};
const plotWindows = [];
let plotIdCounter = 0;

function getSeriesData(name) {
  if (seriesDataCache[name]) return seriesDataCache[name];
  let xs, ys;
  if (name === "altitude") {
    const p = flightData.positions;
    const n = (p.length / 4) | 0;
    xs = new Array(n); ys = new Array(n);
    for (let i = 0; i < n; i++) { xs[i] = p[i*4]; ys[i] = p[i*4 + 3]; }
  } else if (name === "ground_speed") {
    const arr = flightData.speeds;
    const n = (arr.length / 2) | 0;
    xs = new Array(n); ys = new Array(n);
    for (let i = 0; i < n; i++) { xs[i] = arr[i*2]; ys[i] = arr[i*2 + 1]; }
  } else {  // roll, pitch, yaw — sparse, insert null gaps between bursts
    const o = flightData.orientations;
    const t = o.times, v = o[name];
    xs = []; ys = [];
    for (let i = 0; i < t.length; i++) {
      if (i > 0 && t[i] - t[i-1] > ORIENT_GAP_THRESH) {
        xs.push((t[i-1] + t[i]) / 2);
        ys.push(null);
      }
      xs.push(t[i]); ys.push(v[i]);
    }
  }
  seriesDataCache[name] = { xs, ys };
  return seriesDataCache[name];
}

/** Merge multiple (xs, ys) series into uPlot's [xMaster, ys1, ys2, ...] form. */
function buildUplotData(seriesNames) {
  if (seriesNames.length === 0) return [[]];
  const sources = seriesNames.map(getSeriesData);
  const xSet = new Set();
  for (const s of sources) for (const x of s.xs) xSet.add(x);
  const xMerged = Array.from(xSet).sort((a, b) => a - b);
  const out = [xMerged];
  for (const s of sources) {
    const map = new Map();
    for (let i = 0; i < s.xs.length; i++) map.set(s.xs[i], s.ys[i]);
    const ys = new Array(xMerged.length);
    for (let i = 0; i < xMerged.length; i++) {
      const v = map.get(xMerged[i]);
      ys[i] = v === undefined ? null : v;
    }
    out.push(ys);
  }
  return out;
}

const plotResizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const win = plotWindows.find(w => w.body === entry.target);
    if (!win || !win.uplot) continue;
    const w = entry.contentRect.width, h = entry.contentRect.height;
    if (w > 0 && h > 0) win.uplot.setSize({ width: w, height: h });
  }
});

function initPlotPalette() {
  const chipsEl = document.getElementById("plot-palette-chips");
  for (const [name, info] of Object.entries(PLOT_SERIES)) {
    const chip = document.createElement("div");
    chip.className = "plot-chip";
    chip.draggable = true;
    chip.dataset.series = name;
    chip.textContent = info.label;
    chip.style.borderColor = info.color;
    chip.style.color = info.color;
    chip.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", name);
      e.dataTransfer.effectAllowed = "copy";
    });
    chipsEl.appendChild(chip);
  }
  const dropZone = document.getElementById("plot-drop-zone");
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const name = e.dataTransfer.getData("text/plain");
    if (name && PLOT_SERIES[name]) createPlotWindow([name]);
  });
}

function createPlotWindow(initialSeries) {
  const id = ++plotIdCounter;
  const el = document.createElement("div");
  el.className = "plot-window";
  el.dataset.id = id;
  el.innerHTML = `
    <div class="plot-window-header">
      <div class="plot-window-legend"></div>
      <button class="plot-window-close" title="Remove plot">×</button>
    </div>
    <div class="plot-window-body">
      <div class="plot-window-cursor"></div>
    </div>
  `;
  const win = {
    id, el,
    body:   el.querySelector(".plot-window-body"),
    cursor: el.querySelector(".plot-window-cursor"),
    legend: el.querySelector(".plot-window-legend"),
    series: [],
    uplot: null,
  };

  el.addEventListener("dragover", (e) => { e.preventDefault(); el.classList.add("dragover"); });
  el.addEventListener("dragleave", (e) => { if (!el.contains(e.relatedTarget)) el.classList.remove("dragover"); });
  el.addEventListener("drop", (e) => {
    e.preventDefault(); e.stopPropagation();
    el.classList.remove("dragover");
    const name = e.dataTransfer.getData("text/plain");
    if (name && PLOT_SERIES[name] && !win.series.includes(name)) {
      win.series.push(name);
      rebuildWindowChart(win);
    }
  });

  el.querySelector(".plot-window-close").addEventListener("click", () => removePlotWindow(win));

  document.getElementById("plot-windows").appendChild(el);
  plotWindows.push(win);
  plotResizeObserver.observe(win.body);
  ensurePlotAreaExpanded();

  for (const name of initialSeries) {
    if (!win.series.includes(name)) win.series.push(name);
  }
  rebuildWindowChart(win);
  return win;
}

function rebuildWindowChart(win) {
  win.legend.innerHTML = "";
  for (const name of win.series) {
    const info = PLOT_SERIES[name];
    const tag = document.createElement("span");
    tag.className = "plot-legend-item";
    tag.style.color = info.color;
    tag.innerHTML = `<span class="plot-legend-dot" style="background:${info.color}"></span>${info.label}<button class="plot-legend-remove" title="Remove">×</button>`;
    tag.querySelector(".plot-legend-remove").addEventListener("click", () => {
      win.series = win.series.filter(s => s !== name);
      if (win.series.length === 0) removePlotWindow(win);
      else rebuildWindowChart(win);
    });
    win.legend.appendChild(tag);
  }

  if (win.uplot) { win.uplot.destroy(); win.uplot = null; }
  if (win.series.length === 0) return;

  const data = buildUplotData(win.series);
  const w = Math.max(50, win.body.clientWidth);
  const h = Math.max(50, win.body.clientHeight);
  win.initialScales = null;
  const opts = {
    width: w, height: h,
    cursor:  { show: false },
    legend:  { show: false },
    scales:  { x: { time: false } },
    axes: [
      { stroke: "#888", grid: { stroke: "#262626" }, ticks: { stroke: "#444" }, font: "11px system-ui" },
      { stroke: "#888", grid: { stroke: "#262626" }, ticks: { stroke: "#444" }, font: "11px system-ui", size: 50 },
    ],
    series: [
      { label: "t" },
      ...win.series.map(name => ({
        label:  PLOT_SERIES[name].label,
        stroke: PLOT_SERIES[name].color,
        width: 1.5,
        spanGaps: false,
        points: { show: false },
      })),
    ],
    hooks: {
      ready: [(u) => {
        win.initialScales = {
          x: { min: u.scales.x.min, max: u.scales.x.max },
          y: { min: u.scales.y.min, max: u.scales.y.max },
        };
        if (win.xRange == null) {
          win.xRange = win.initialScales.x.max - win.initialScales.x.min;
        }
        // Snap to a view centered on the current playback time.
        if (viewer3D && viewer3D.clock) {
          const e = Cesium.JulianDate.secondsDifference(
            viewer3D.clock.currentTime, viewer3D.clock.startTime);
          u.setScale("x", { min: e - win.xRange / 2, max: e + win.xRange / 2 });
        }
      }],
    },
  };
  win.uplot = new uPlot(opts, data, win.body);
  attachZoomHandlers(win);
}

function removePlotWindow(win) {
  if (win.uplot) { win.uplot.destroy(); win.uplot = null; }
  plotResizeObserver.unobserve(win.body);
  win.el.remove();
  const i = plotWindows.indexOf(win);
  if (i >= 0) plotWindows.splice(i, 1);
}

/** First time a window is added, expand the plot area to a sensible default. */
function ensurePlotAreaExpanded() {
  const area = document.getElementById("plot-area");
  if (area.offsetHeight < 80) area.style.height = "260px";
}

/** Single drag handle at the top of the plot area resizes the whole pane. */
function initPlotAreaResize() {
  const handle = document.getElementById("plot-area-resize");
  const area = document.getElementById("plot-area");
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    const startY = e.clientY;
    const startH = area.offsetHeight;
    const maxH = Math.floor(window.innerHeight * 0.85);
    const onMove = (ev) => {
      const h = Math.max(36, Math.min(maxH, startH - (ev.clientY - startY)));
      area.style.height = h + "px";
      if (lmap) lmap.invalidateSize();
      if (viewer3D && viewer3D.resize) viewer3D.resize();
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

/**
 * Wheel zoom for "follow time" mode:
 *  - Wheel on plot area  → zoom X-window + Y (X anchored at the centerline,
 *                          Y anchored at the mouse cursor)
 *  - Wheel on x axis     → shrink/grow the visible time window
 *  - Wheel on y axis     → zoom Y only (anchored at mouse y)
 *  - Double-click        → reset window to full data extent and y to initial
 */
function attachZoomHandlers(win) {
  const u = win.uplot;
  const overEl  = u.over;
  const xAxisEl = u.root.querySelector(".u-axis.u-axis-x") || u.root.querySelectorAll(".u-axis")[0];
  const yAxisEl = u.root.querySelector(".u-axis.u-axis-y") || u.root.querySelectorAll(".u-axis")[1];

  // Fallback initial-scale capture if the ready hook hasn't fired yet.
  if (!win.initialScales || win.initialScales.x.min == null) {
    if (u.scales.x.min != null && u.scales.y.min != null) {
      win.initialScales = {
        x: { min: u.scales.x.min, max: u.scales.x.max },
        y: { min: u.scales.y.min, max: u.scales.y.max },
      };
      if (win.xRange == null) win.xRange = win.initialScales.x.max - win.initialScales.x.min;
    }
  }

  function getElapsed() {
    if (!viewer3D || !viewer3D.clock) return 0;
    return Cesium.JulianDate.secondsDifference(
      viewer3D.clock.currentTime, viewer3D.clock.startTime);
  }

  /** Apply the current xRange centered on `elapsed`. */
  function applyXWindow() {
    if (win.xRange == null) return;
    const e = getElapsed();
    u.setScale("x", { min: e - win.xRange / 2, max: e + win.xRange / 2 });
    placeCursorCenter(win);
  }

  /** Y-axis zoom anchored at a data-y value. Clamped to the y data extent. */
  function zoomY(factor, anchor) {
    const sc = u.scales.y;
    const init = win.initialScales && win.initialScales.y;
    if (sc.min == null || sc.max == null || !Number.isFinite(anchor)) return;
    let newMin = anchor - (anchor - sc.min) * factor;
    let newMax = anchor + (sc.max - anchor) * factor;
    if (init && init.min != null) {
      const fullRange = init.max - init.min;
      if (newMax - newMin >= fullRange) {
        newMin = init.min; newMax = init.max;
      } else {
        if (newMin < init.min) { newMax += init.min - newMin; newMin = init.min; }
        if (newMax > init.max) { newMin -= newMax - init.max; newMax = init.max; }
      }
    }
    if (newMin < newMax) u.setScale("y", { min: newMin, max: newMax });
  }

  /** X-window zoom: scale the visible time window, keeping it centered on `elapsed`. */
  function zoomXWindow(factor) {
    if (win.xRange == null) return;
    const init = win.initialScales && win.initialScales.x;
    let newRange = win.xRange * factor;
    if (init && init.min != null) {
      const fullRange = init.max - init.min;
      if (newRange > fullRange) newRange = fullRange;
    }
    if (newRange < 0.1) newRange = 0.1;   // ~100 ms minimum window
    win.xRange = newRange;
    applyXWindow();
  }

  function anchorYVal(e) {
    const r = overEl.getBoundingClientRect();
    return u.posToVal(e.clientY - r.top, "y");
  }

  const ZOOM_K = 0.0015;

  overEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = Math.exp(e.deltaY * ZOOM_K);
    zoomXWindow(factor);
    zoomY(factor, anchorYVal(e));
  }, { passive: false });

  if (xAxisEl) xAxisEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomXWindow(Math.exp(e.deltaY * ZOOM_K));
  }, { passive: false });

  if (yAxisEl) yAxisEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomY(Math.exp(e.deltaY * ZOOM_K), anchorYVal(e));
  }, { passive: false });

  overEl.addEventListener("dblclick", () => {
    const init = win.initialScales;
    if (!init || init.x.min == null || init.y.min == null) return;
    win.xRange = init.x.max - init.x.min;
    applyXWindow();
    u.setScale("y", init.y);
  });
}

/**
 * Position the playback cursor at the visual horizontal center of the plot
 * drawing area. The chart slides so the current playback time is at this
 * centerline, so the cursor itself never moves (relative to the chart frame).
 */
function placeCursorCenter(w) {
  if (!w.uplot) { w.cursor.style.display = "none"; return; }
  const overRect = w.uplot.over.getBoundingClientRect();
  if (overRect.width <= 0) { w.cursor.style.display = "none"; return; }
  const bodyRect = w.body.getBoundingClientRect();
  w.cursor.style.display = "block";
  w.cursor.style.left = ((overRect.left - bodyRect.left) + overRect.width / 2) + "px";
}

/**
 * Per-tick update: slide each chart's x-range so the current playback time
 * sits at the cursor (center), then re-center the cursor itself.
 */
function updatePlotCursors(elapsedSec) {
  for (const w of plotWindows) {
    if (!w.uplot || w.xRange == null) {
      if (w.cursor) w.cursor.style.display = "none";
      continue;
    }
    const half = w.xRange / 2;
    const sc = w.uplot.scales.x;
    const newMin = elapsedSec - half;
    const newMax = elapsedSec + half;
    if (sc.min !== newMin || sc.max !== newMax) {
      w.uplot.setScale("x", { min: newMin, max: newMax });
    }
    placeCursorCenter(w);
  }
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

    /* Plot cursors */
    updatePlotCursors(elapsed);
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
  // Heading and speed computed from consecutive GPS positions — more reliable
  // than the CSV columns, which can be stale or zero during sparse GPS fixes.
  headingSampled = buildHeadingFromPositions(flightData.positions, epochJulian);
  speedSampled   = buildSpeedFromPositions(flightData.positions, epochJulian);

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

  /* Plot palette (PlotJuggler-style draggable variables) */
  initPlotPalette();
  initPlotAreaResize();

  /* Start playing */
  viewer3D.clock.shouldAnimate = true;
  document.getElementById("play-btn").innerHTML = "&#9646;&#9646;";

  setStatus("");
}
