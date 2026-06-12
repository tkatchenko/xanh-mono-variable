#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

try:
    from fontTools.designspaceLib import AxisDescriptor, DesignSpaceDocument, RuleDescriptor, SourceDescriptor
    from fontTools.otlLib.builder import buildStatTable
    from fontTools.ttLib import TTFont, newTable
    from fontTools.varLib import build as build_variable_font
except ModuleNotFoundError as exc:
    raise SystemExit(
        "FontTools is required. On this machine, run with "
        "/opt/homebrew/opt/python@3.9/bin/python3.9 source/build_variable.py"
    ) from exc


Point = Tuple[int, int]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def repo_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def resolve_repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def point_tuple(point: Any) -> Point:
    return int(round(point[0])), int(round(point[1]))


def polygon_area(points: Sequence[Point]) -> float:
    area = 0.0
    for index, current in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += current[0] * nxt[1] - nxt[0] * current[1]
    return area / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            x_intersection = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < x_intersection:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def contour_centre(points: Sequence[Point]) -> Point:
    return (
        int(round(sum(point[0] for point in points) / len(points))),
        int(round(sum(point[1] for point in points) / len(points))),
    )


def contour_embolden_metadata(contours: Sequence[Sequence[Point]]) -> List[Tuple[int, bool]]:
    areas = [polygon_area(contour) for contour in contours]
    metadata = []
    for index, contour in enumerate(contours):
        area = areas[index]
        contour_sign = 1 if area >= 0 else -1
        centre = contour_centre(contour)
        containers = [
            container_index
            for container_index, container in enumerate(contours)
            if container_index != index
            and abs(areas[container_index]) > abs(area)
            and point_in_polygon(centre, container)
        ]
        if containers:
            parent_index = max(containers, key=lambda container_index: abs(areas[container_index]))
            metadata.append((1 if areas[parent_index] >= 0 else -1, True))
        else:
            metadata.append((contour_sign, False))
    return metadata


def unit_normal(dx: float, dy: float, outward_sign: int) -> Tuple[float, float]:
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    if outward_sign > 0:
        return dy / length, -dx / length
    return -dy / length, dx / length


def neighbouring_point(points: Sequence[Point], index: int, direction: int) -> Point:
    current = points[index]
    count = len(points)
    for offset in range(1, count + 1):
        candidate = points[(index + direction * offset) % count]
        if candidate != current:
            return candidate
    return current


def is_on_curve(flags: Sequence[int], index: int) -> bool:
    return bool(flags[index] & 0x01)


def touches_curve(flags: Sequence[int], contour_indices: Sequence[int], local_index: int) -> bool:
    glyph_index = contour_indices[local_index]
    prev_index = contour_indices[(local_index - 1) % len(contour_indices)]
    next_index = contour_indices[(local_index + 1) % len(contour_indices)]
    return (
        not is_on_curve(flags, glyph_index)
        or not is_on_curve(flags, prev_index)
        or not is_on_curve(flags, next_index)
    )


def embolden_simple_glyph(
    glyph: Any,
    glyf: Any,
    strength: float,
    curve_strength_multiplier: float,
    counter_strength_multiplier: float,
) -> None:
    coordinates, end_points, flags = glyph.getCoordinates(glyf)
    if not coordinates or not end_points:
        return

    new_coordinates = list(point_tuple(point) for point in coordinates)
    contours = []
    start = 0
    for end in end_points:
        contours.append([point_tuple(coordinates[index]) for index in range(start, end + 1)])
        start = end + 1

    contour_metadata = contour_embolden_metadata(contours)
    start = 0
    for contour_index, end in enumerate(end_points):
        contour_indices = list(range(start, end + 1))
        contour_points = contours[contour_index]
        if len(contour_points) < 3:
            start = end + 1
            continue

        outward_sign, is_counter = contour_metadata[contour_index]

        for local_index, glyph_index in enumerate(contour_indices):
            point = contour_points[local_index]
            prev_point = neighbouring_point(contour_points, local_index, -1)
            next_point = neighbouring_point(contour_points, local_index, 1)

            prev_dx = point[0] - prev_point[0]
            prev_dy = point[1] - prev_point[1]
            next_dx = next_point[0] - point[0]
            next_dy = next_point[1] - point[1]

            prev_normal = unit_normal(prev_dx, prev_dy, outward_sign)
            next_normal = unit_normal(next_dx, next_dy, outward_sign)
            nx = prev_normal[0] + next_normal[0]
            ny = prev_normal[1] + next_normal[1]
            normal_length = math.hypot(nx, ny)
            if normal_length == 0:
                nx, ny = next_normal
                normal_length = math.hypot(nx, ny)
            if normal_length == 0:
                continue

            point_strength = strength
            if touches_curve(flags, contour_indices, local_index):
                point_strength *= curve_strength_multiplier
            if is_counter:
                point_strength *= counter_strength_multiplier

            scale = point_strength / normal_length
            new_coordinates[glyph_index] = (
                int(round(point[0] + nx * scale)),
                int(round(point[1] + ny * scale)),
            )

        start = end + 1

    coordinates[:] = new_coordinates
    glyph.coordinates = coordinates
    glyph.endPtsOfContours = end_points
    glyph.flags = flags
    glyph.recalcBounds(glyf)


def update_simple_sidebearings(font: TTFont) -> None:
    glyf = font["glyf"]
    hmtx = font["hmtx"]
    for glyph_name in font.getGlyphOrder():
        if glyph_name not in hmtx.metrics:
            continue
        glyph = glyf[glyph_name]
        if glyph.isComposite():
            glyph.recalcBounds(glyf)
        advance, _ = hmtx.metrics[glyph_name]
        left_sidebearing = getattr(glyph, "xMin", 0)
        hmtx.metrics[glyph_name] = (advance, left_sidebearing)


def make_bold_font(
    font: TTFont,
    strength: float,
    curve_strength_multiplier: float,
    counter_strength_multiplier: float,
) -> TTFont:
    bold = copy.deepcopy(font)
    glyf = bold["glyf"]
    for glyph_name in bold.getGlyphOrder():
        glyph = glyf[glyph_name]
        if glyph.isComposite():
            continue
        embolden_simple_glyph(glyph, glyf, strength, curve_strength_multiplier, counter_strength_multiplier)

    update_simple_sidebearings(bold)
    bold["OS/2"].usWeightClass = 700
    bold["head"].macStyle |= 1
    return bold


def remap_composite_components(glyph: Any, mapping: Mapping[str, str]) -> None:
    if not glyph.isComposite():
        return
    for component in glyph.components:
        if component.glyphName in mapping:
            component.glyphName = mapping[component.glyphName]


def rename_glyph_references(value: Any, mapping: Mapping[str, str], seen: set[int] | None = None) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = rename_glyph_references(item, mapping, seen)
        return value
    if isinstance(value, tuple):
        return tuple(rename_glyph_references(item, mapping, seen) for item in value)
    if isinstance(value, dict):
        new_items = {}
        for key, item in value.items():
            new_key = rename_glyph_references(key, mapping, seen)
            new_items[new_key] = rename_glyph_references(item, mapping, seen)
        value.clear()
        value.update(new_items)
        return value
    if not hasattr(value, "__dict__"):
        return value

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return value
    seen.add(value_id)

    for attr, item in list(value.__dict__.items()):
        if attr.startswith("_"):
            continue
        setattr(value, attr, rename_glyph_references(item, mapping, seen))
    return value


def merge_gdef(base: TTFont, italic: TTFont, mapping: Mapping[str, str]) -> None:
    if "GDEF" not in base or "GDEF" not in italic:
        return
    base_table = base["GDEF"].table
    italic_table = italic["GDEF"].table

    for attr in ("GlyphClassDef", "MarkAttachClassDef"):
        base_def = getattr(base_table, attr, None)
        italic_def = getattr(italic_table, attr, None)
        if base_def is None or italic_def is None:
            continue
        for glyph_name, class_value in italic_def.classDefs.items():
            if glyph_name in mapping:
                base_def.classDefs[mapping[glyph_name]] = class_value


def merge_layout_table(base: TTFont, italic: TTFont, tag: str, mapping: Mapping[str, str]) -> None:
    if tag not in base or tag not in italic:
        return

    base_table = base[tag].table
    italic_table = copy.deepcopy(italic[tag].table)
    rename_glyph_references(italic_table, mapping)

    if not base_table.LookupList or not italic_table.LookupList:
        return

    lookup_offset = len(base_table.LookupList.Lookup)
    base_table.LookupList.Lookup.extend(italic_table.LookupList.Lookup)
    base_table.LookupList.LookupCount = len(base_table.LookupList.Lookup)

    base_features = base_table.FeatureList.FeatureRecord
    italic_features = italic_table.FeatureList.FeatureRecord
    if len(base_features) != len(italic_features):
        raise ValueError(f"{tag} feature count differs between roman and italic")

    for base_feature, italic_feature in zip(base_features, italic_features):
        if base_feature.FeatureTag != italic_feature.FeatureTag:
            raise ValueError(f"{tag} feature order differs at {base_feature.FeatureTag}")
        additions = [index + lookup_offset for index in italic_feature.Feature.LookupListIndex]
        base_feature.Feature.LookupListIndex.extend(additions)
        base_feature.Feature.LookupCount = len(base_feature.Feature.LookupListIndex)


def add_italic_alternates(base: TTFont, italic: TTFont, suffix: str = ".ital") -> TTFont:
    merged = copy.deepcopy(base)
    base_order = merged.getGlyphOrder()
    mapping = {glyph_name: f"{glyph_name}{suffix}" for glyph_name in base_order if glyph_name != ".notdef"}
    alternate_order = [mapping[glyph_name] for glyph_name in base_order if glyph_name in mapping]

    glyf = merged["glyf"]
    hmtx = merged["hmtx"]
    italic_glyf = italic["glyf"]
    italic_hmtx = italic["hmtx"]

    for glyph_name in base_order:
        if glyph_name not in mapping:
            continue
        alternate_name = mapping[glyph_name]
        alternate_glyph = copy.deepcopy(italic_glyf[glyph_name])
        remap_composite_components(alternate_glyph, mapping)
        glyf.glyphs[alternate_name] = alternate_glyph
        hmtx.metrics[alternate_name] = italic_hmtx.metrics[glyph_name]

    merged.setGlyphOrder(base_order + alternate_order)
    merged["maxp"].numGlyphs = len(merged.getGlyphOrder())
    merge_gdef(merged, italic, mapping)
    merge_layout_table(merged, italic, "GPOS", mapping)
    merge_layout_table(merged, italic, "GSUB", mapping)

    return merged


def clear_and_set_name(font: TTFont, name_id: int, value: str) -> None:
    name_table = font["name"]
    name_table.names = [name for name in name_table.names if name.nameID != name_id]
    name_table.setName(value, name_id, 3, 1, 0x409)
    name_table.setName(value, name_id, 1, 0, 0)


def set_names(font: TTFont, family_name: str, postscript_prefix: str) -> None:
    clear_and_set_name(font, 1, family_name)
    clear_and_set_name(font, 2, "Regular")
    clear_and_set_name(font, 3, f"3.101;NONE;{postscript_prefix}")
    clear_and_set_name(font, 4, family_name)
    clear_and_set_name(font, 5, "Version 3.101")
    clear_and_set_name(font, 6, postscript_prefix)
    clear_and_set_name(font, 16, family_name)
    clear_and_set_name(font, 17, "Regular")
    clear_and_set_name(font, 25, postscript_prefix)


def set_stat_table(font: TTFont) -> None:
    buildStatTable(
        font,
        [
            {
                "tag": "wght",
                "name": "Weight",
                "ordering": 0,
                "values": [
                    {"value": 400, "name": "Regular", "flags": 0x2, "linkedValue": 700},
                    {"value": 700, "name": "Bold"},
                ],
            },
            {
                "tag": "ital",
                "name": "Italic",
                "ordering": 1,
                "values": [
                    {"value": 0, "name": "Roman", "flags": 0x2, "linkedValue": 1},
                    {"value": 1, "name": "Italic"},
                ],
            },
        ],
        elidedFallbackName="Regular",
    )


def make_designspace(
    path: Path,
    regular_master: Path,
    bold_master: Path,
    glyph_order: Sequence[str],
    weight_axis: Mapping[str, Any],
    italic_axis: Mapping[str, Any],
    italic_suffix: str,
) -> None:
    document = DesignSpaceDocument()

    weight = AxisDescriptor()
    weight.name = "Weight"
    weight.tag = "wght"
    weight.minimum = weight_axis["minimum"]
    weight.default = weight_axis["default"]
    weight.maximum = weight_axis["maximum"]
    weight.map = [(weight.minimum, weight.minimum), (weight.maximum, weight.maximum)]
    weight.labelNames["en"] = "Weight"
    document.addAxis(weight)

    italic = AxisDescriptor()
    italic.name = "Italic"
    italic.tag = "ital"
    italic.minimum = italic_axis["minimum"]
    italic.default = italic_axis["default"]
    italic.maximum = italic_axis["maximum"]
    italic.map = [(italic.minimum, italic.minimum), (italic.maximum, italic.maximum)]
    italic.labelNames["en"] = "Italic"
    document.addAxis(italic)

    regular_source = SourceDescriptor()
    regular_source.name = "Regular"
    regular_source.path = str(regular_master)
    regular_source.familyName = "Xanh Mono Variable"
    regular_source.styleName = "Regular"
    regular_source.location = {"Weight": weight.default, "Italic": italic.default}
    regular_source.copyInfo = True
    regular_source.copyLib = True
    regular_source.copyFeatures = True
    document.addSource(regular_source)

    bold_source = SourceDescriptor()
    bold_source.name = "Bold"
    bold_source.path = str(bold_master)
    bold_source.familyName = "Xanh Mono Variable"
    bold_source.styleName = "Bold"
    bold_source.location = {"Weight": weight.maximum, "Italic": italic.default}
    document.addSource(bold_source)

    substitutions = [
        (glyph_name, f"{glyph_name}{italic_suffix}")
        for glyph_name in glyph_order
        if glyph_name != ".notdef"
    ]
    rule = RuleDescriptor()
    rule.name = "Italic alternates"
    rule.conditionSets = [[
        {
            "name": "Italic",
            "minimum": italic_axis["substitution_threshold"],
            "maximum": italic.maximum,
        }
    ]]
    rule.subs = substitutions
    document.addRule(rule)

    path.parent.mkdir(parents=True, exist_ok=True)
    document.write(path)


def save_woff(font_path: Path, output_path: Path) -> None:
    font = TTFont(font_path)
    font.flavor = "woff"
    font.save(output_path)


def run_checked(args: Sequence[str], cwd: Path) -> None:
    try:
        subprocess.run(args, cwd=str(cwd), check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing command: {args[0]}") from exc


def write_css(path: Path, family_name: str, woff_path: Path, woff2_path: Path, ttf_path: Path) -> None:
    css = f'''@font-face {{
  font-family: "{family_name}";
  src:
    url("../fonts/webfonts/{woff2_path.name}") format("woff2-variations"),
    url("../fonts/webfonts/{woff_path.name}") format("woff-variations"),
    url("../fonts/ttf/{ttf_path.name}") format("truetype-variations");
  font-weight: 400 700;
  font-style: normal;
  font-display: swap;
}}

@font-face {{
  font-family: "{family_name}";
  src:
    url("../fonts/webfonts/{woff2_path.name}") format("woff2-variations"),
    url("../fonts/webfonts/{woff_path.name}") format("woff-variations"),
    url("../fonts/ttf/{ttf_path.name}") format("truetype-variations");
  font-weight: 400 700;
  font-style: italic;
  font-display: swap;
}}

.xanh-mono-variable {{
  font-family: "{family_name}", ui-monospace, "SFMono-Regular", Consolas, monospace;
  font-weight: 400;
  font-style: normal;
  font-variation-settings: "wght" 400, "ital" 0;
}}

.xanh-mono-variable-bold {{
  font-family: "{family_name}", ui-monospace, "SFMono-Regular", Consolas, monospace;
  font-weight: 700;
  font-style: normal;
  font-variation-settings: "wght" 700, "ital" 0;
}}

.xanh-mono-variable-italic {{
  font-family: "{family_name}", ui-monospace, "SFMono-Regular", Consolas, monospace;
  font-weight: 400;
  font-style: italic;
  font-variation-settings: "wght" 400, "ital" 1;
}}

.xanh-mono-variable-bold-italic {{
  font-family: "{family_name}", ui-monospace, "SFMono-Regular", Consolas, monospace;
  font-weight: 700;
  font-style: italic;
  font-variation-settings: "wght" 700, "ital" 1;
}}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(css, encoding="utf-8")


def write_build_readme(path: Path) -> None:
    text = """Xanh Mono Variable

Variable build of Xanh Mono.

Axes:
wght 400-700
ital 0-1

Defaults:
wght 400
ital 0

Bold:
wght 700

Build:
sh source/build.sh

Outputs:
fonts/ttf/XanhMonoVariable.ttf
fonts/otf/XanhMonoVariable.otf
fonts/webfonts/XanhMonoVariable.woff
fonts/webfonts/XanhMonoVariable.woff2
build/css/xanh-mono-variable.css
build/specimen/XanhMonoVariable-specimen.pdf

Test:
range-test.html
"""
    path.write_text(text, encoding="utf-8")


def write_specimen_html(path: Path, family_name: str, woff2_path: Path, sample_text: str) -> None:
    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{family_name} specimen</title>
<style>
@font-face {{
  font-family: "{family_name}";
  src: url("../fonts/webfonts/{woff2_path.name}") format("woff2-variations");
  font-weight: 400 700;
  font-style: normal italic;
}}
body {{
  margin: 48px;
  font-family: "{family_name}", monospace;
  color: #111;
}}
h1 {{
  font-size: 36px;
  font-weight: 700;
  margin: 0 0 24px;
}}
section {{
  margin: 0 0 28px;
}}
.label {{
  font-size: 14px;
  font-weight: 400;
  margin-bottom: 8px;
}}
.sample {{
  font-size: 42px;
  line-height: 1.25;
  margin: 0;
}}
.small {{
  font-size: 18px;
  line-height: 1.5;
}}
</style>
</head>
<body>
<h1>{family_name}</h1>
<section>
  <div class="label">Axes: wght 400 to 700, ital 0 to 1. Default: wght 400, ital 0.</div>
  <p class="small">Family: {family_name}<br>Formats: TTF, OTF, WOFF, WOFF2, CSS.</p>
</section>
<section>
  <div class="label">Regular. font-weight: 400; font-style: normal; "wght" 400, "ital" 0.</div>
  <p class="sample" style="font-weight:400; font-style:normal; font-variation-settings:'wght' 400,'ital' 0;">{sample_text}</p>
</section>
<section>
  <div class="label">Bold. font-weight: 700; font-style: normal; "wght" 700, "ital" 0.</div>
  <p class="sample" style="font-weight:700; font-style:normal; font-variation-settings:'wght' 700,'ital' 0;">{sample_text}</p>
</section>
<section>
  <div class="label">Italic. font-weight: 400; font-style: italic; "wght" 400, "ital" 1.</div>
  <p class="sample" style="font-weight:400; font-style:italic; font-variation-settings:'wght' 400,'ital' 1;">{sample_text}</p>
</section>
<section>
  <div class="label">Bold Italic. font-weight: 700; font-style: italic; "wght" 700, "ital" 1.</div>
  <p class="sample" style="font-weight:700; font-style:italic; font-variation-settings:'wght' 700,'ital' 1;">{sample_text}</p>
</section>
</body>
</html>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_specimen_pdf(output_path: Path, font_path: Path, family_name: str, sample_text: str, temp_dir: Path) -> None:
    hb_view = shutil.which("hb-view")
    pdfunite = shutil.which("pdfunite")
    if not hb_view or not pdfunite:
        raise SystemExit("hb-view and pdfunite are required to build the specimen PDF")

    pages = [
        ("overview", "wght=400,ital=0", f"{family_name}\nAxes: wght 400-700, ital 0-1\nDefault: wght 400, ital 0"),
        ("regular", "wght=400,ital=0", f"Regular\nwght=400 ital=0\n{sample_text}\nABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n0123456789"),
        ("bold", "wght=700,ital=0", f"Bold\nwght=700 ital=0\n{sample_text}\nABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n0123456789"),
        ("italic", "wght=400,ital=1", f"Italic\nwght=400 ital=1\n{sample_text}\nABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n0123456789"),
        ("bold-italic", "wght=700,ital=1", f"Bold Italic\nwght=700 ital=1\n{sample_text}\nABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n0123456789"),
    ]

    page_paths = []
    for name, variations, text in pages:
        text_path = temp_dir / f"{name}.txt"
        pdf_path = temp_dir / f"{name}.pdf"
        text_path.write_text(text, encoding="utf-8")
        run_checked(
            [
                hb_view,
                str(font_path),
                "--text-file",
                str(text_path),
                "--output-file",
                str(pdf_path),
                "--output-format",
                "pdf",
                "--font-size",
                "40",
                "--line-space",
                "16",
                "--margin",
                "72",
                "--variations",
                variations,
            ],
            temp_dir,
        )
        page_paths.append(pdf_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked([pdfunite, *[str(path) for path in page_paths], str(output_path)], temp_dir)


def validate_font(path: Path, regular_source: TTFont, italic_source: TTFont) -> List[str]:
    font = TTFont(path)
    issues = []
    for table in ("fvar", "gvar", "GSUB", "GPOS", "GDEF", "STAT"):
        if table not in font:
            issues.append(f"missing {table}")
    axes = {axis.axisTag: axis for axis in font["fvar"].axes}
    for tag in ("wght", "ital"):
        if tag not in axes:
            issues.append(f"missing {tag} axis")

    for glyph_name in regular_source.getGlyphOrder():
        if glyph_name == ".notdef":
            continue
        if font["hmtx"].metrics[glyph_name][0] != regular_source["hmtx"].metrics[glyph_name][0]:
            issues.append(f"{glyph_name} advance differs from regular source")
            break
        italic_name = f"{glyph_name}.ital"
        if font["hmtx"].metrics[italic_name][0] != italic_source["hmtx"].metrics[glyph_name][0]:
            issues.append(f"{italic_name} advance differs from italic source")
            break
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="source/build-variable.config.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    root = repo_root_from_config(config_path)
    config = load_config(config_path)

    family_name = config["family_name"]
    postscript_prefix = config["postscript_prefix"]
    regular_path = resolve_repo_path(root, config["input_regular_ttf"])
    italic_path = resolve_repo_path(root, config["input_italic_ttf"])
    output_root = resolve_repo_path(root, config["output_root"])
    weight_axis = config["weight_axis"]
    italic_axis = config["italic_axis"]
    sample_text = config["sample_text"]

    build_fonts = output_root / "fonts"
    build_ttf_dir = build_fonts / "ttf"
    build_otf_dir = build_fonts / "otf"
    build_web_dir = build_fonts / "webfonts"
    build_css_dir = output_root / "css"
    build_specimen_dir = output_root / "specimen"
    build_master_dir = output_root / "masters"

    for directory in (build_ttf_dir, build_otf_dir, build_web_dir, build_css_dir, build_specimen_dir, build_master_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    regular = TTFont(regular_path)
    italic = TTFont(italic_path)
    if regular.getGlyphOrder() != italic.getGlyphOrder():
        raise SystemExit("Regular and italic glyph order must match")

    set_names(regular, family_name, postscript_prefix)
    set_names(italic, family_name, postscript_prefix)

    strength = float(weight_axis["bold_strength_units"])
    curve_strength_multiplier = float(weight_axis.get("curve_strength_multiplier", 1.0))
    counter_strength_multiplier = float(weight_axis.get("counter_strength_multiplier", 1.0))
    regular_bold = make_bold_font(regular, strength, curve_strength_multiplier, counter_strength_multiplier)
    italic_bold = make_bold_font(italic, strength, curve_strength_multiplier, counter_strength_multiplier)

    set_names(regular_bold, family_name, postscript_prefix)
    set_names(italic_bold, family_name, postscript_prefix)

    merged_regular = add_italic_alternates(regular, italic)
    merged_bold = add_italic_alternates(regular_bold, italic_bold)
    set_names(merged_regular, family_name, postscript_prefix)
    set_names(merged_bold, family_name, postscript_prefix)

    regular_master_path = build_master_dir / f"{postscript_prefix}-RegularMaster.ttf"
    bold_master_path = build_master_dir / f"{postscript_prefix}-BoldMaster.ttf"
    merged_regular.save(regular_master_path)
    merged_bold.save(bold_master_path)

    designspace_path = build_master_dir / f"{postscript_prefix}.designspace"
    make_designspace(
        designspace_path,
        regular_master_path,
        bold_master_path,
        regular.getGlyphOrder(),
        weight_axis,
        italic_axis,
        ".ital",
    )

    variable_font, _, _ = build_variable_font(str(designspace_path))
    set_names(variable_font, family_name, postscript_prefix)
    set_stat_table(variable_font)
    variable_font["OS/2"].usWeightClass = int(weight_axis["default"])
    variable_font["OS/2"].usWidthClass = 5
    variable_font["head"].macStyle = 0
    variable_font["post"].italicAngle = 0

    ttf_output = build_ttf_dir / f"{postscript_prefix}.ttf"
    otf_output = build_otf_dir / f"{postscript_prefix}.otf"
    woff_output = build_web_dir / f"{postscript_prefix}.woff"
    woff2_output = build_web_dir / f"{postscript_prefix}.woff2"
    css_output = build_css_dir / "xanh-mono-variable.css"
    specimen_html = build_specimen_dir / f"{postscript_prefix}-specimen.html"
    specimen_pdf = build_specimen_dir / f"{postscript_prefix}-specimen.pdf"

    variable_font.save(ttf_output)
    shutil.copy2(ttf_output, otf_output)

    if config.get("build_webfonts", True):
        save_woff(ttf_output, woff_output)
        run_checked(["woff2_compress", str(ttf_output)], root)
        compressed_path = ttf_output.with_suffix(".woff2")
        if compressed_path.exists():
            shutil.move(str(compressed_path), str(woff2_output))

    write_css(css_output, family_name, woff_output, woff2_output, ttf_output)
    write_build_readme(output_root / "README.txt")
    write_specimen_html(specimen_html, family_name, woff2_output, sample_text)

    if config.get("build_specimen_pdf", True):
        with tempfile.TemporaryDirectory(prefix="xanh-specimen-") as temp_name:
            write_specimen_pdf(specimen_pdf, ttf_output, family_name, sample_text, Path(temp_name))

    issues = validate_font(ttf_output, regular, italic)
    if issues:
        raise SystemExit("Validation failed: " + "; ".join(issues))

    if config.get("replace_font_dirs", False):
        target_dirs = {
            root / "fonts" / "ttf": [ttf_output],
            root / "fonts" / "otf": [otf_output],
            root / "fonts" / "webfonts": [woff_output, woff2_output],
        }
        for target_dir, outputs in target_dirs.items():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            for output in outputs:
                shutil.copy2(output, target_dir / output.name)

    print(f"Built {ttf_output}")
    print(f"Built {woff2_output}")
    print(f"Built {specimen_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
