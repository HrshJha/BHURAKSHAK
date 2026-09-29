"""Build the fully editable BHURAKSHAK diagrams.net architecture and QA preview.

All boxes and connectors are native mxGraph objects. The SVG is only a preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
import textwrap
import xml.etree.ElementTree as ET


OUT = Path(__file__).parent
W, H = 2560, 1440
NAVY = "#10264A"
BLUE = "#2878AF"
PURPLE = "#6952A3"
GREEN = "#2D8B71"
RISK = "#CF6D38"
GREY = "#66778D"


@dataclass(frozen=True)
class Box:
    id: str
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: tuple[str, ...] = ()
    kind: str = "built"  # built | planned | gated | future | neutral | panel | tag
    fs: int = 18
    title_fs: int = 19
    align: str = "left"


@dataclass(frozen=True)
class Arrow:
    id: str
    source: str
    target: str
    points: tuple[tuple[int, int], ...]
    color: str = BLUE
    dashed: bool = False
    label: str = ""
    label_at: tuple[int, int] | None = None
    width: int = 3


boxes: list[Box] = []
arrows: list[Arrow] = []


def b(id, x, y, w, h, title, *lines, kind="built", fs=18, title_fs=19, align="left"):
    boxes.append(Box(id, x, y, w, h, title, tuple(lines), kind, fs, title_fs, align))


def a(id, source, target, *points, color=BLUE, dashed=False, label="", label_at=None, width=3):
    arrows.append(Arrow(id, source, target, tuple(points), color, dashed, label, label_at, width))


# Canvas and layer frames.
b("header", 44, 24, 1600, 88, "BHURAKSHAK", "End-to-end technical architecture · SIH 2026 / PS 26025", kind="tag", fs=23, title_fs=38)
b("legend", 1730, 28, 790, 90, "STATUS KEY", "Purple solid = implemented code   ·   Dashed blue = planned integration   ·   Dashed gold = gated   ·   Dashed grey = future", kind="neutral", fs=17, title_fs=17)

for id, x, w, title, subtitle in [
    ("A", 40, 395, "A · PHYSICAL MODEL & SENSORS", "Single-trapdoor prototype configuration"),
    ("B", 455, 360, "B · WIRELESS & EDGE", "Target runtime · integration pending"),
    ("C", 835, 925, "C · DATA + LIVE ML INFERENCE", "Implemented model logic · Pi deployment pending"),
    ("D", 1780, 350, "D · BACKEND & STORAGE", "Planned local services"),
    ("E", 2150, 370, "E · DASHBOARD & OPERATOR", "Planned application integration"),
]:
    b(f"panel_{id}", x, 150, w, 835, title, subtitle, kind="panel", fs=16, title_fs=21)

# A: no exact node positions or M8 pitch are inferred from the older PDF.
b("trapdoor", 60, 235, 355, 215, "SINGLE-TRAPDOOR MODEL", "One controlled surface opening", "Node positions / screw pitch TBD", kind="planned", fs=17, title_fs=20)
b("surface_left", 85, 352, 103, 17, "", kind="neutral", title_fs=1)
b("surface_right", 293, 352, 97, 17, "", kind="neutral", title_fs=1)
b("single_door", 188, 355, 105, 40, "ONE TRAPDOOR", kind="neutral", title_fs=14, align="center")
b("actuator", 224, 397, 18, 32, "", kind="neutral", title_fs=1)
b("actuator_label", 253, 398, 142, 35, "ACTUATOR / KNOB", kind="neutral", title_fs=13)
b("esp32", 60, 480, 355, 260, "ESP32 SENSOR NODE × 4", "MPU9250 · tilt, acceleration, vibration", "VL53L0X · ToF / crack-opening change", "Ultrasonic · vertical displacement", "On-node vibration RMS/peak summary", kind="planned", fs=18, title_fs=20)
b("ground_truth", 60, 770, 355, 165, "MECHANICAL GROUND TRUTH", "Independent reference movement", "Actuator count + trial ID", "Training / validation only", kind="planned", fs=18, title_fs=20)

# B: physical networking is explicitly outside the repo's Data+ML scope.
b("lora", 475, 250, 320, 165, "LoRa SENSOR TRANSPORT", "ID / time / summary packet", "RSSI / SNR link health", "Mesh / ACKs pending", kind="planned", fs=18, title_fs=20)
b("gateway", 475, 455, 320, 375, "RASPBERRY PI 5 EDGE GATEWAY", "LoRa reception + packet decoding", "Python ingestion + local buffer", "MQTT only if configured", "Target host for features + inference", kind="planned", fs=18, title_fs=20)
b("packet_fields", 490, 697, 290, 128, "PACKET FIELDS", "node_id · timestamp", "tilt · vibration_rms / peak", "tof_distance · displacement", "battery · RSSI · SNR", kind="planned", fs=14, title_fs=16)
b("edge_first", 475, 865, 320, 72, "EDGE-FIRST TARGET", "No internet in core-loop design", kind="planned", fs=16, title_fs=19)

# C: the code path is implemented; it has not been joined to a live Pi feed.
b("quality", 855, 245, 275, 188, "DATA QUALITY & SYNC", "Missing / duplicate packets", "Time alignment + drift", "Corruption + baseline checks", "Fault ≠ ground movement", fs=16, title_fs=19)
b("window", 1150, 245, 270, 188, "WINDOWING", "60 steps · stride 10", "Same v2 schema in train + inference", "Mean, std, slope, rates, persistence", fs=18, title_fs=19)
b("features", 1440, 245, 300, 188, "FEATURE GROUPS", "Physical: tilt, displacement", "Temporal: mean, slope, rates", "Vibration: RMS, peak, bands", "Health + physics residual", fs=16, title_fs=19)
b("iforest", 855, 475, 270, 160, "ISOLATION FOREST", "Healthy rows only", "Anomaly score output", "Not subsidence probability", fs=16, title_fs=20)
b("physics", 1145, 475, 270, 160, "PHYSICS CONSISTENCY", "Expected vs observed", "Residual = observed − expected", "Plausibility check", fs=16, title_fs=20)
b("spatial", 1435, 475, 305, 160, "SPATIAL COHERENCE", "Neighbour fusion gated", "DGPS / InSAR optional", "Current events: one node", kind="gated", fs=16, title_fs=19)
b("xgb", 895, 670, 400, 150, "XGBOOST · RISK CLASSIFIER", "Features + score + residual", "NORMAL / WARNING / CRITICAL", "Current 3-class model", fs=17, title_fs=21)
b("calibration", 1320, 670, 350, 150, "ISOTONIC CALIBRATION", "Validation-fitted class probabilities", "Development calibration ≠ field validation", fs=18, title_fs=20)
b("explain", 855, 850, 330, 108, "MODEL EXPLAINABILITY", "Contributions · physical signals · model version", fs=17, title_fs=18)
b("alert", 1210, 850, 530, 108, "RISK DECISION & ALERT STATE MACHINE", "GREEN  →  WATCH  →  WARNING  →  CRITICAL", "Persistence + evidence; test reached GREEN / WATCH", fs=17, title_fs=19)

# D: separate planned integration from code already in src/risk.
b("db", 1800, 250, 310, 245, "LOCAL DATABASE · PLANNED", "SQLite MVP target", "Readings · features · predictions", "Alerts · trials · model metadata", "Offline persistence", kind="planned", fs=18, title_fs=19)
b("api", 1800, 540, 310, 250, "FASTAPI REST · PLANNED", "Node status + latest readings", "Risk map + alerts + history", "Model explanation payload", "Local service between edge and UI", kind="planned", fs=18, title_fs=19)
b("cloud", 1800, 840, 310, 118, "OPTIONAL CLOUD SYNC", "Buffer, then sync when online", "PostgreSQL at scale", kind="future", fs=17, title_fs=19)

# E: these are specified capabilities, not a deployed integration.
b("dashboard", 2170, 250, 330, 430, "BHURAKSHAK GIS DASHBOARD", "React / Next.js + Leaflet target", kind="planned", fs=17, title_fs=18)
b("dash_map", 2185, 350, 145, 130, "GIS RISK MAP", "Locations", "Risk overlay", kind="planned", fs=15, title_fs=16)
b("dash_series", 2345, 350, 140, 130, "TIME SERIES", "Live sensors", "Deformation", kind="planned", fs=15, title_fs=16)
b("dash_alerts", 2185, 495, 145, 130, "RISK + ALERTS", "Four states", "History", kind="planned", fs=15, title_fs=16)
b("dash_explain", 2345, 495, 140, 130, "EXPLAIN / HEALTH", "Contributions", "Node status", kind="planned", fs=15, title_fs=16)
b("operator", 2170, 720, 330, 135, "MINE OPERATOR", "View risk and inspect evidence", "Acknowledge / override via planned UI", kind="planned", fs=18, title_fs=20)
b("notify", 2170, 885, 330, 75, "OPTIONAL NOTIFICATIONS", "SMS · email · mobile · siren", kind="future", fs=17, title_fs=18)

# Offline lane: dataset provenance is explicit. Generated tabletop stand-ins
# from the older two-door plan are not treated as measured one-door trials.
b("offline_title", 40, 1005, 2080, 70, "OFFLINE ML TRAINING + VALIDATION", "Separate from live inference · frozen artifacts are promoted only after evaluation", kind="panel", fs=17, title_fs=22)
b("synthetic", 58, 1090, 290, 172, "PHYSICS-COUPLED DATA", "Latent deformation → coupled signals", "Noise, faults, scenarios + manifest", "Implemented corpus", fs=17, title_fs=19)
b("trials", 368, 1090, 275, 172, "SINGLE-DOOR TRIALS", "Signal + independent reference", "Trial metadata + descent", "Physical collection pending", kind="planned", fs=17, title_fs=19)
b("store", 665, 1090, 215, 172, "FEATURE STORE", "Validate · window", "Schema v2 + provenance", fs=18, title_fs=19)
b("split", 900, 1090, 220, 172, "GROUPED SPLIT", "Event / parameter families", "Train · validation · locked test", "No window leakage", fs=17, title_fs=19)
b("train_if", 1140, 1080, 235, 78, "TRAIN ISOLATION FOREST", "Normal-only rows", fs=16, title_fs=16)
b("train_xgb", 1140, 1183, 235, 78, "TRAIN XGBOOST", "Three-class supervised", fs=16, title_fs=17)
b("eval", 1400, 1090, 315, 172, "EVALUATION", "Critical recall · macro PR-AUC/F1", "Normal false alarms · lead time", "Calibration + grouped holdout", fs=17, title_fs=19)
b("registry", 1740, 1090, 350, 172, "MODEL REGISTRY + ARTIFACTS", "Frozen .joblib + checksums", "Model · feature schema · dataset version", "Pi deployment: planned integration", fs=17, title_fs=19)
b("evidence", 58, 1290, 2032, 70, "LOCKED SYNTHETIC TEST", "Tuned XGBoost: 9.33% Critical recall · median lead time −3.33 h · useful early warning not demonstrated", kind="tag", fs=18, title_fs=18)

b("future_panel", 2145, 1005, 375, 355, "FUTURE / RESEARCH", "Not part of the essential MVP flow", kind="panel", fs=17, title_fs=22)
b("future_forecast", 2165, 1080, 335, 72, "DEFORMATION FORECASTING", "TCN dev model; GRU/LSTM research", kind="future", fs=16, title_fs=17)
b("future_geo", 2165, 1160, 335, 55, "SATELLITE / DGPS FUSION", kind="future", title_fs=16)
b("future_gnn", 2165, 1222, 335, 55, "GNN SPATIAL MODEL", kind="future", title_fs=16)
b("future_field", 2165, 1284, 335, 55, "FIELD-DATA RETRAINING", kind="future", title_fs=16)

# Main live data path. Edges are independent mxCells, never baked into shapes.
a("e_a1", "trapdoor", "esp32", (238, 450), (238, 480))
a("e_actuate", "actuator", "single_door", (233, 397), (233, 395), color=GREY, width=2)
a("e_a2", "trapdoor", "ground_truth", (115, 450), (115, 770), color=GREY, dashed=True, label="reference only", label_at=(122, 750))
a("e_ab", "esp32", "lora", (415, 575), (445, 575), (445, 330), (475, 330), label="sensor summaries", label_at=(447, 445), dashed=True)
a("e_b1", "lora", "gateway", (635, 415), (635, 455), dashed=True)
a("e_bc", "gateway", "quality", (795, 635), (825, 635), (825, 340), (855, 340), dashed=True, label="packet stream", label_at=(824, 605))
a("e_c1", "quality", "window", (1130, 338), (1150, 338), color=PURPLE)
a("e_c2", "window", "features", (1420, 338), (1440, 338), color=PURPLE)
a("e_c3", "features", "iforest", (1490, 433), (1490, 450), (990, 450), (990, 475), color=PURPLE, label="v2 features", label_at=(1150, 447))
a("e_c4", "features", "physics", (1580, 433), (1580, 458), (1280, 458), (1280, 475), color=PURPLE)
a("e_c5", "features", "spatial", (1665, 433), (1665, 475), color=PURPLE, dashed=True)
a("e_c_direct", "features", "xgb", (1440, 405), (1425, 405), (1425, 660), (1290, 660), (1290, 670), color=PURPLE, label="feature vector", label_at=(1335, 657))
a("e_c6", "iforest", "xgb", (990, 635), (990, 670), color=PURPLE, label="anomaly score", label_at=(1000, 650))
a("e_c7", "physics", "xgb", (1280, 635), (1280, 650), (1190, 650), (1190, 670), color=PURPLE, label="residual", label_at=(1220, 646))
a("e_c8", "spatial", "xgb", (1515, 635), (1515, 650), (1280, 650), (1280, 670), color=PURPLE, dashed=True, label="gated", label_at=(1440, 645))
a("e_c9", "xgb", "calibration", (1295, 745), (1320, 745), color=PURPLE)
a("e_c10", "xgb", "explain", (1000, 820), (1000, 850), color=PURPLE)
a("e_c11", "calibration", "alert", (1500, 820), (1500, 850), color=RISK, label="calibrated P", label_at=(1510, 836))
a("e_cd", "alert", "api", (1740, 905), (1770, 905), (1770, 665), (1800, 665), color=RISK, dashed=True, label="state + evidence", label_at=(1770, 820))
a("e_explain_api", "explain", "api", (1050, 958), (1050, 973), (1775, 973), (1775, 740), (1800, 740), color=PURPLE, dashed=True, label="explanation", label_at=(1490, 969))
a("e_db", "api", "db", (1955, 540), (1955, 495), color=GREEN, dashed=True, label="read / write", label_at=(1965, 520))
a("e_db_return", "db", "api", (2025, 495), (2025, 540), color=GREEN, dashed=True)
a("e_dc", "db", "cloud", (2110, 390), (2125, 390), (2125, 899), (2110, 899), color=GREY, dashed=True)
a("e_de", "api", "dashboard", (2110, 660), (2140, 660), (2140, 450), (2170, 450), color=GREEN, dashed=True, label="REST", label_at=(2145, 640))
a("e_e1", "dashboard", "operator", (2335, 680), (2335, 720), color=GREEN, dashed=True, label="view / acknowledge", label_at=(2340, 700))
a("e_e1_return", "operator", "dashboard", (2200, 720), (2200, 680), color=GREEN, dashed=True)
a("e_e_feedback", "dashboard", "api", (2170, 620), (2135, 620), (2135, 770), (2110, 770), color=GREEN, dashed=True, label="ack / override", label_at=(2140, 748))
a("e_e2", "dashboard", "notify", (2475, 680), (2510, 680), (2510, 920), (2500, 920), color=GREEN, dashed=True)
a("e_quality_flags", "quality", "db", (1010, 245), (1010, 226), (1860, 226), (1860, 250), color=GREY, dashed=True, label="data-quality flags · planned persistence", label_at=(1500, 220), width=2)
a("e_features_db", "features", "db", (1740, 385), (1765, 385), (1765, 315), (1800, 315), color=GREY, dashed=True, label="feature rows", label_at=(1767, 360), width=2)

# Ground truth branches into training only. The bottom lane is deliberately
# disconnected from live classification except for frozen-artifact deployment.
a("e_gt", "ground_truth", "trials", (60, 850), (25, 850), (25, 1083), (505, 1083), (505, 1090), color=GREY, dashed=True, label="reference labels", label_at=(50, 998))
a("e_t1", "synthetic", "store", (348, 1176), (665, 1176), color=PURPLE)
a("e_t2", "trials", "store", (643, 1200), (665, 1200), color=PURPLE, dashed=True)
a("e_t3", "store", "split", (880, 1176), (900, 1176), color=PURPLE)
a("e_t4", "split", "train_if", (1120, 1130), (1140, 1130), color=PURPLE)
a("e_t5", "split", "train_xgb", (1120, 1220), (1140, 1220), color=PURPLE)
a("e_t6", "train_if", "eval", (1375, 1120), (1388, 1120), (1388, 1140), (1400, 1140), color=PURPLE)
a("e_t7", "train_xgb", "eval", (1375, 1220), (1388, 1220), (1388, 1205), (1400, 1205), color=PURPLE)
a("e_t8", "eval", "registry", (1715, 1176), (1740, 1176), color=PURPLE)
a("e_deploy", "registry", "gateway", (1920, 1090), (1920, 990), (805, 990), (805, 795), (795, 795), color=PURPLE, dashed=True, label="frozen .joblib → edge (planned)", label_at=(1200, 985), width=3)


def box_style(box: Box) -> str:
    styles = {
        "built": ("#F7F8FD", "#846CAD", "0"),
        "planned": ("#F8FCFE", "#438DB8", "1"),
        "gated": ("#FFFDF6", "#B29555", "1"),
        "future": ("#FAFAFA", "#A4ABB6", "1"),
        "neutral": ("#FFFFFF", "#C7D1DD", "0"),
        "panel": ("#FBFCFE", "#CDD8E5", "0"),
        "tag": ("#F1F5FA", "#B6C7DA", "0"),
    }
    fill, stroke, dashed = styles[box.kind]
    pad = 14 if box.kind != "panel" else 12
    return (f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth={'2' if box.kind != 'panel' else '1.5'};dashed={dashed};dashPattern=7 5;"
            f"fontColor={NAVY};fontFamily=Arial;fontSize={box.fs};align={box.align};verticalAlign=top;"
            f"spacingTop={pad};spacingLeft={pad};spacingRight={pad};spacingBottom=8;shadow=0;")


def wrap_words(value: str, width: int, fs: int) -> list[str]:
    # Explicit line breaks keep the diagrams.net file legible in versions with
    # slightly different HTML label rendering, and make the preview faithful.
    max_chars = max(12, int((width - 32) / (fs * 0.54)))
    return textwrap.wrap(value, width=max_chars, break_long_words=False, break_on_hyphens=False) or [""]


def box_html(box: Box) -> str:
    title = "<br>".join(escape(s) for s in wrap_words(box.title, box.w, box.title_fs))
    lines = "<br>".join(escape(s) for line in box.lines for s in wrap_words(line, box.w, box.fs))
    if lines:
        return f'<div style="font-size:{box.title_fs}px;font-weight:700;color:{NAVY}">{title}</div><div style="font-size:{box.fs}px;line-height:1.23;margin-top:6px;color:#334762">{lines}</div>'
    return f'<div style="font-size:{box.title_fs}px;font-weight:700;color:{NAVY}">{title}</div>'


def build_drawio() -> None:
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "modified": "2026-09-29T00:00:00.000Z", "agent": "Codex", "version": "24.7.17", "type": "device"})
    diagram = ET.SubElement(mxfile, "diagram", {"name": "Complete architecture", "id": "bhurakshak-end-to-end"})
    model = ET.SubElement(diagram, "mxGraphModel", {"dx": str(W), "dy": str(H), "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(W), "pageHeight": str(H), "math": "0", "shadow": "0"})
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    # Frames first. Connectors second. Components last for clean stacking.
    panel_boxes = [box for box in boxes if box.kind == "panel"]
    content_boxes = [box for box in boxes if box.kind != "panel"]
    for box in panel_boxes:
        cell = ET.SubElement(root, "mxCell", {"id": box.id, "value": box_html(box), "style": box_style(box), "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": str(box.x), "y": str(box.y), "width": str(box.w), "height": str(box.h), "as": "geometry"})

    box_by_id = {box.id: box for box in boxes}
    for arrow in arrows:
        src = box_by_id[arrow.source]
        dst = box_by_id[arrow.target]
        sx, sy = arrow.points[0]
        tx, ty = arrow.points[-1]
        exit_x = max(0, min(1, (sx - src.x) / src.w))
        exit_y = max(0, min(1, (sy - src.y) / src.h))
        entry_x = max(0, min(1, (tx - dst.x) / dst.w))
        entry_y = max(0, min(1, (ty - dst.y) / dst.h))
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
                 f"strokeColor={arrow.color};strokeWidth={arrow.width};endArrow=block;endFill=1;"
                 f"dashed={'1' if arrow.dashed else '0'};dashPattern=8 5;fontColor={arrow.color};"
                 f"exitX={exit_x:.4f};exitY={exit_y:.4f};entryX={entry_x:.4f};entryY={entry_y:.4f};"
                 "exitPerimeter=0;entryPerimeter=0;fontFamily=Arial;fontSize=15;labelBackgroundColor=#FFFFFF;")
        cell = ET.SubElement(root, "mxCell", {"id": arrow.id, "value": arrow.label, "style": style, "edge": "1", "parent": "1", "source": arrow.source, "target": arrow.target})
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geom, "mxPoint", {"x": str(arrow.points[0][0]), "y": str(arrow.points[0][1]), "as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", {"x": str(arrow.points[-1][0]), "y": str(arrow.points[-1][1]), "as": "targetPoint"})
        if len(arrow.points) > 2:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for x, y in arrow.points[1:-1]:
                ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})

    for box in content_boxes:
        cell = ET.SubElement(root, "mxCell", {"id": box.id, "value": box_html(box), "style": box_style(box), "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": str(box.x), "y": str(box.y), "width": str(box.w), "height": str(box.h), "as": "geometry"})

    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(OUT / "BHURAKSHAK_Complete_Architecture.drawio", encoding="utf-8", xml_declaration=True)


def svg_text(x: int, y: int, box: Box) -> str:
    titles = wrap_words(box.title, box.w, box.title_fs)
    out = []
    for i, title in enumerate(titles):
        out.append(f'<text x="{x}" y="{y + i * (box.title_fs + 5)}" font-family="Arial,Helvetica,sans-serif" fill="{NAVY}" font-size="{box.title_fs}" font-weight="700">{escape(title)}</text>')
    base = y + len(titles) * (box.title_fs + 5) + 7
    body = [s for line in box.lines for s in wrap_words(line, box.w, box.fs)]
    for i, line in enumerate(body):
        out.append(f'<text x="{x}" y="{base + i * (box.fs + 7)}" font-family="Arial,Helvetica,sans-serif" fill="#334762" font-size="{box.fs}">{escape(line)}</text>')
    return "".join(out)


def build_preview() -> None:
    fills = {"built": ("#F7F8FD", "#846CAD"), "planned": ("#F8FCFE", "#438DB8"), "gated": ("#FFFDF6", "#B29555"), "future": ("#FAFAFA", "#A4ABB6"), "neutral": ("#FFFFFF", "#C7D1DD"), "panel": ("#FBFCFE", "#CDD8E5"), "tag": ("#F1F5FA", "#B6C7DA")}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><rect width="{W}" height="{H}" fill="white"/>']
    for color in {e.color for e in arrows}:
        key = color.replace("#", "")
        svg.append(f'<defs><marker id="arr{key}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="{color}"/></marker></defs>')
    for box in [x for x in boxes if x.kind == "panel"]:
        fill, stroke = fills[box.kind]
        svg.append(f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        svg.append(svg_text(box.x + 15, box.y + 31, box))
    for arrow in arrows:
        pts = " ".join(f"{x},{y}" for x, y in arrow.points)
        dash = ' stroke-dasharray="9 6"' if arrow.dashed else ""
        key = arrow.color.replace("#", "")
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{arrow.color}" stroke-width="{arrow.width}"{dash} marker-end="url(#arr{key})"/>')
        if arrow.label and arrow.label_at:
            x, y = arrow.label_at
            svg.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" fill="{arrow.color}" font-size="15" font-weight="700" stroke="white" stroke-width="5" paint-order="stroke">{escape(arrow.label)}</text>')
    for box in [x for x in boxes if x.kind != "panel"]:
        fill, stroke = fills[box.kind]
        dash = ' stroke-dasharray="9 6"' if box.kind in {"planned", "gated", "future"} else ""
        svg.append(f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        svg.append(svg_text(box.x + 15, box.y + 31, box))
    svg.append("</svg>")
    (OUT / "BHURAKSHAK_Complete_Architecture_preview.svg").write_text("".join(svg))


def build_mermaid() -> None:
    # This source is deliberately hand-structured rather than derived from
    # coordinates, so it remains readable and easy to paste into Mermaid Live.
    mermaid = r'''flowchart LR
  %% Solid: implemented Data+ML code. Dashed: planned/gated/future integration.
  subgraph A["A · PHYSICAL MODEL & SENSOR HARDWARE"]
    TD["Single trapdoor<br/>controlled opening · measured descent"]
    ESP["ESP32 nodes ×4<br/>MPU9250 · VL53L0X · ultrasonic<br/>on-node vibration RMS/peak"]
    GT["Independent mechanical ground truth<br/>movement · actuator count · trial ID"]
    TD -.-> ESP
    TD -.->|reference measurements| GT
  end
  subgraph B["B · WIRELESS COMMUNICATION & EDGE"]
    LORA["LoRa transport<br/>node ID · timestamp · RSSI/SNR<br/>mesh/ACK integration pending"]
    PI["Raspberry Pi 5 target<br/>packet decode · Python ingest<br/>local buffer · optional MQTT"]
    PKT["Packet fields<br/>tilt · vibration_rms/peak · ToF<br/>displacement · battery · RSSI/SNR"]
    EDGE["Edge-first target<br/>core loop operates without internet"]
    LORA -.-> PI
    PI --- PKT
    PI -.-> EDGE
  end
  subgraph C["C · DATA PROCESSING & LIVE ML INFERENCE"]
    DQ["Data quality & sync<br/>gaps · duplicates · drift · fault flags"]
    WIN["Windowing<br/>60 steps / stride 10 · schema v2"]
    FEAT["Physical · temporal · vibration<br/>health · physics features"]
    IF["Isolation Forest<br/>healthy rows only → anomaly score"]
    PHYS["Physics consistency<br/>observed − expected → residual"]
    GEO["GATED spatial / DGPS / InSAR<br/>no validated multi-node fusion"]
    XGB["XGBoost risk classifier<br/>NORMAL / WARNING / CRITICAL"]
    CAL["Isotonic probability calibration"]
    EXPL["Model explanation<br/>contributions · physical signals · version"]
    ALERT["Risk decision / alert state machine<br/>GREEN → WATCH → WARNING → CRITICAL<br/>persistence + evidence"]
    DQ --> WIN --> FEAT
    FEAT --> IF -->|anomaly score| XGB
    FEAT --> PHYS -->|physics residual| XGB
    FEAT -.-> GEO -.-> XGB
    FEAT --> XGB
    XGB --> CAL --> ALERT
    XGB --> EXPL
  end
  subgraph D["D · BACKEND, STORAGE & DEPLOYMENT · PLANNED"]
    API["FastAPI REST<br/>nodes · readings · risk · alerts<br/>history · explanation"]
    DB["Local SQLite MVP target<br/>readings · features · predictions<br/>alerts · metadata"]
    CLOUD["Optional cloud sync<br/>after internet restoration<br/>PostgreSQL/TimescaleDB at scale"]
    API -.->|read/write| DB
    DB -.->|query results| API
    DB -.-> CLOUD
  end
  subgraph E["E · DASHBOARD, ALERTS & OPERATOR · PLANNED"]
    UI["React/Next.js + Leaflet target<br/>risk map · live readings · time series<br/>four-state panel · alerts · explanation · health"]
    MAP["GIS risk map"]
    SERIES["Live sensor / deformation series"]
    RISKUI["Four-state risk + alert history"]
    HEALTH["Explanation + node health"]
    OP["Mine operator<br/>inspect · acknowledge · permitted override"]
    NOTIF["Optional SMS / email / mobile / siren"]
    UI --- MAP
    UI --- SERIES
    UI --- RISKUI
    UI --- HEALTH
    UI -.-> OP
    OP -.->|acknowledge / override| UI
    UI -.-> NOTIF
  end
  ESP -.->|summary payload| LORA
  PI -.->|packets| DQ
  DQ -.->|quality flags| DB
  FEAT -.->|feature rows| DB
  ALERT -.->|state + evidence| API
  EXPL -.->|explanation| API
  API -.->|REST| UI
  UI -.->|acknowledgement / override| API

  subgraph OFF["OFFLINE ML TRAINING & VALIDATION · separate workflow"]
    SYN["Physics-coupled synthetic corpus<br/>latent deformation · faults · manifest"]
    EXP["Single-trapdoor physical trials<br/>independent reference · pending"]
    STORE["Preprocessing + feature store<br/>schema v2"]
    SPLIT["Leakage-safe split<br/>event / parameter families"]
    TIF["Train Isolation Forest<br/>normal-only"]
    TXGB["Train XGBoost<br/>three-class supervised"]
    EVAL["Held-out evaluation<br/>Critical recall · macro PR-AUC/F1<br/>false alarms · lead time · calibration"]
    LOCKED["Locked synthetic test<br/>tuned XGBoost Critical recall 9.33%<br/>median lead time −3.33 h; useful warning unproven"]
    REG["Registry + frozen .joblib artifacts<br/>model / schema / dataset versions"]
    SYN --> STORE
    EXP -.-> STORE
    STORE --> SPLIT
    SPLIT --> TIF --> EVAL
    SPLIT --> TXGB --> EVAL
    EVAL --> REG
    EVAL --> LOCKED
  end
  GT -.->|training labels only| EXP
  REG -.->|planned edge deployment| PI

  subgraph FUT["FUTURE / RESEARCH"]
    TCN["TCN dev model; GRU/LSTM research<br/>forecast displacement or velocity"]
    SAT["Expanded Sentinel-1 / DGPS fusion"]
    GNN["GNN spatial modelling"]
    FIELD["Field-data retraining"]
  end

  classDef built fill:#F7F8FD,stroke:#846CAD,color:#10264A,stroke-width:2px;
  classDef planned fill:#F8FCFE,stroke:#438DB8,color:#10264A,stroke-width:2px,stroke-dasharray:7 5;
  classDef gated fill:#FFFDF6,stroke:#B29555,color:#10264A,stroke-width:2px,stroke-dasharray:7 5;
  classDef future fill:#FAFAFA,stroke:#A4ABB6,color:#10264A,stroke-width:2px,stroke-dasharray:7 5;
  class DQ,WIN,FEAT,IF,PHYS,XGB,CAL,EXPL,ALERT,SYN,STORE,SPLIT,TIF,TXGB,EVAL,REG,LOCKED built;
  class TD,ESP,GT,LORA,PI,PKT,EDGE,API,DB,UI,MAP,SERIES,RISKUI,HEALTH,OP,EXP planned;
  class GEO gated;
  class CLOUD,NOTIF,TCN,SAT,GNN,FIELD future;
'''
    (OUT / "BHURAKSHAK_Complete_Architecture.mmd").write_text(mermaid)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_drawio()
    build_mermaid()
    build_preview()
    print(f"Created {len(boxes)} editable boxes and {len(arrows)} connectors")
