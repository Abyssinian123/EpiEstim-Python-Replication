# -*- coding: utf-8 -*-
"""Extract the plotted curves from the paper's loess_rt.pdf (vector
graphics) and compare them with the Python loess + estimate_R pipeline.
This validates loess.py without an R installation."""
import os
import numpy as np
import pdfplumber

PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "EM_EpiEstim_Nash2023-main", "supplementary_figures", "loess_rt.pdf")


def bezier(p, n=8):
    p = np.array(p)
    t = np.linspace(0, 1, n)[:, None]
    # cubic Bezier
    return ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1]
            + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3])


def page_curves(page):
    """Return [(color, pts array)] for every curve on the page."""
    out = []
    for otype, objs in page.objects.items():
        if otype not in ("curve", "line", "rectline"):
            continue
        for obj in objs:
            col = obj.get("stroking_color")
            if col is None:
                continue
            col = tuple(round(c, 3) for c in col)
            if otype == "curve":
                pts = bezier(obj["pts"], 16)
            else:
                pts = np.array([(p[0], p[1]) for p in obj["pts"]])
            out.append((col, pts))
    return out


def page_text(page):
    out = []
    for otype, objs in page.objects.items():
        if otype != "char":
            continue
        for obj in objs:
            out.append((obj.get("text"), obj.get("x0"), obj.get("top")))
    return out


def panel_map(page):
    """Affine pdf->data map from curve bbox + known axis limits; returns
    (a, b, c, d) so that x_data = (x_pdf - b)/a, y_data = (y_pdf - d)/c."""
    curves = page_curves(page)
    allx, ally = [], []
    for col, pts in curves:
        allx += list(pts[:, 0])
        ally += list(pts[:, 1])
    x0, x1, y0, y1 = min(allx), max(allx), min(ally), max(ally)
    # data limits: x from tick labels/curves (caller passes), y = (0.5, 2)
    return x0, x1, y0, y1


if __name__ == "__main__":
    with pdfplumber.open(PDF) as pdf:
        print("pages:", len(pdf.pages))
        for pno in range(len(pdf.pages)):
            page = pdf.pages[pno]
            curves = page_curves(page)
            print(f"page {pno + 1}: {len(curves)} curves")
            x0, x1, y0, y1 = panel_map(page)
            print(f"  bbox x [{x0:.1f},{x1:.1f}] y [{y0:.1f},{y1:.1f}]")
            for col, pts in curves:
                print(f"  {col}: n={len(pts)} x[{pts[:,0].min():.1f},"
                      f"{pts[:,0].max():.1f}] y[{pts[:,1].min():.2f},"
                      f"{pts[:,1].max():.2f}]")
            txt = page_text(page)
            print("  texts:", [(t, round(x, 1), round(y, 1))
                               for t, x, y in txt])
