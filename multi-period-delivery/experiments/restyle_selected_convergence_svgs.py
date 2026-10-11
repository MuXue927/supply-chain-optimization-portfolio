#!/usr/bin/env python3
"""
Batch restyle selected convergence SVGs for thesis usage.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def ns(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def iter_with_parent(root: ET.Element):
    stack = [(None, root)]
    while stack:
        parent, node = stack.pop()
        yield parent, node
        for child in list(node):
            stack.append((node, child))


def comment_texts(node: ET.Element) -> List[str]:
    out: List[str] = []
    for ch in node.iter():
        if not isinstance(ch.tag, str):
            t = (ch.text or "").strip()
            if t:
                out.append(t)
    return out


def has_comment(node: ET.Element, marker: str) -> bool:
    m = marker.lower()
    return any(m in t.lower() for t in comment_texts(node))


def remove_nodes(root: ET.Element, predicate) -> None:
    doomed = []
    for parent, node in iter_with_parent(root):
        if parent is None:
            continue
        if predicate(node):
            doomed.append((parent, node))
    for parent, node in doomed:
        try:
            parent.remove(node)
        except Exception:
            pass


def parse_plot_box(
    axes_node: ET.Element,
) -> Optional[Tuple[float, float, float, float]]:
    patch = axes_node.find("./" + ns("g") + "[@id='patch_2']/" + ns("path"))
    if patch is None:
        return None
    d = patch.attrib.get("d", "")
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", d)]
    if len(nums) < 8:
        return None
    x0, y1, x1, _y1b, _x1b, y0 = nums[:6]
    return x0, x1, y0, y1


def expand_svg_canvas(
    root: ET.Element,
    *,
    left: float = 72.0,
    right: float = 220.0,
    top: float = 74.0,
    bottom: float = 8.0,
) -> None:
    view_box = root.attrib.get("viewBox", "")
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", view_box)]
    if len(nums) == 4:
        x, y, w, h = nums
        root.set(
            "viewBox",
            f"{x - left:.3f} {y - top:.3f} {w + left + right:.3f} {h + top + bottom:.3f}",
        )

    def _expand_pt(attr: str, inc: float) -> None:
        val = root.attrib.get(attr, "")
        if not val.endswith("pt"):
            return
        try:
            num = float(val[:-2])
            root.set(attr, f"{num + inc:.3f}pt")
        except Exception:
            return

    _expand_pt("width", left + right)
    _expand_pt("height", top + bottom)


def make_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    font: str = "SimSun",
    rotate: Optional[Tuple[float, float, float]] = None,
) -> ET.Element:
    t = ET.Element(
        ns("text"),
        {
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "style": f"font-family: '{font}'; font-size: {size}px; fill: #000000;",
        },
    )
    if rotate is not None:
        ang, rx, ry = rotate
        t.set("transform", f"rotate({ang:.1f} {rx:.3f} {ry:.3f})")
    t.text = text
    return t


def add_custom_legend(container: ET.Element, x_left: float, y_top: float) -> None:
    g = ET.Element(ns("g"), {"id": "legend_custom"})
    lx = x_left
    ly = y_top
    lw = 132.0
    lh = 46.0

    ET.SubElement(
        g,
        ns("rect"),
        {
            "x": f"{lx:.3f}",
            "y": f"{ly:.3f}",
            "width": f"{lw:.3f}",
            "height": f"{lh:.3f}",
            "style": "fill: #ffffff; stroke: #cccccc; stroke-width: 0.8;",
        },
    )

    ET.SubElement(
        g,
        ns("line"),
        {
            "x1": f"{(lx + 10):.3f}",
            "y1": f"{(ly + 16):.3f}",
            "x2": f"{(lx + 34):.3f}",
            "y2": f"{(ly + 16):.3f}",
            "style": "stroke: #1f77b4; stroke-width: 2;",
        },
    )
    g.append(make_text(lx + 40, ly + 20, "搜索目标", size=11, font="SimSun"))

    ET.SubElement(
        g,
        ns("line"),
        {
            "x1": f"{(lx + 10):.3f}",
            "y1": f"{(ly + 34):.3f}",
            "x2": f"{(lx + 34):.3f}",
            "y2": f"{(ly + 34):.3f}",
            "style": "stroke: #d62728; stroke-width: 2;",
        },
    )
    g.append(make_text(lx + 40, ly + 38, "实际目标", size=11, font="SimSun"))
    container.append(g)


def restyle_svg(input_path: Path, output_path: Path) -> None:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(input_path, parser=parser)
    root = tree.getroot()

    # Remove default legend.
    remove_nodes(
        root,
        lambda n: isinstance(n.tag, str)
        and n.tag == ns("g")
        and n.attrib.get("id") in {"legend_1", "legend_custom"},
    )

    # Remove title and old axis-label text groups only.
    def is_old_label_group(n: ET.Element) -> bool:
        if not isinstance(n.tag, str) or n.tag != ns("g"):
            return False
        gid = n.attrib.get("id", "")
        if not gid.startswith("text_"):
            return False
        return (
            has_comment(n, "objective convergence (search vs actual)")
            or has_comment(n, "iteration (#)")
            or has_comment(n, "search objective")
            or has_comment(n, "actual objective")
        )

    remove_nodes(root, is_old_label_group)

    # Remove dashed grid lines.
    remove_nodes(
        root,
        lambda n: isinstance(n.tag, str)
        and n.tag == ns("path")
        and "stroke-dasharray" in n.attrib.get("style", "")
        and "#b0b0b0" in n.attrib.get("style", "").lower(),
    )

    # Remove raw line groups (keep smoothed two curves).
    remove_nodes(
        root,
        lambda n: isinstance(n.tag, str)
        and n.tag == ns("g")
        and (
            has_comment(n, "search objective (raw)")
            or has_comment(n, "actual objective (raw)")
        ),
    )

    axes = root.find("./" + ns("g") + "[@id='figure_1']/" + ns("g") + "[@id='axes_1']")
    fig = root.find("./" + ns("g") + "[@id='figure_1']")
    if axes is not None:
        box = parse_plot_box(axes)
        if box is not None:
            x0, x1, y0, y1 = box
            axes.append(
                make_text(
                    (x0 + x1) / 2.0, y1 + 30.0, "迭代次数", size=15, font="SimSun"
                )
            )
            axes.append(
                make_text(
                    x0 - 56.0,
                    (y0 + y1) / 2.0,
                    "搜索目标值",
                    size=14,
                    font="SimSun",
                    rotate=(-90.0, x0 - 56.0, (y0 + y1) / 2.0),
                )
            )
            axes.append(
                make_text(
                    x1 + 58.0,
                    (y0 + y1) / 2.0,
                    "实际目标值",
                    size=14,
                    font="SimSun",
                    rotate=(90.0, x1 + 58.0, (y0 + y1) / 2.0),
                )
            )
            # Place legend above plotting rectangle in top margin, fully outside axes/ticks.
            if fig is not None:
                add_custom_legend(fig, x1 - 132.0, y0 - 58.0)

    # Expand canvas to avoid clipping rotated y-axis labels.
    expand_svg_canvas(root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def restyle_folder(input_dir: Path, output_dir: Path) -> Tuple[int, int]:
    svgs = sorted(p for p in input_dir.glob("*_Objective.svg") if p.is_file())
    ok = 0
    for src in svgs:
        dst = output_dir / src.name
        try:
            restyle_svg(src, dst)
            ok += 1
        except Exception:
            pass
    return ok, len(svgs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Restyle selected convergence SVGs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="ExperimentsResults/thesis_analysis/representative_convergence_curves",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ExperimentsResults/thesis_analysis/representative_convergence_curves_styled",
    )
    args = parser.parse_args()

    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ok, total = restyle_folder(input_dir, output_dir)
    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"styled={ok}")
    print(f"total={total}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
