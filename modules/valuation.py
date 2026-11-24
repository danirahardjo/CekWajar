import math
import numpy as np


def calculate_fair_value(
    current_price: float,
    eps: float,
    bvps: float,
    fair_per: float,
    fair_pbv: float,
) -> dict:
    """Menghitung harga wajar menggunakan Graham Number, Relative PER, dan Relative PBV."""
    models = {}

    if eps > 0 and bvps > 0:
        graham = math.sqrt(22.5 * eps * bvps)
        models["Graham Number"] = graham

    if eps > 0:
        models[f"Valuasi PER ({fair_per:.1f}x)"] = eps * fair_per

    if bvps > 0:
        models[f"Valuasi PBV ({fair_pbv:.1f}x)"] = bvps * fair_pbv

    if models:
        composite_fair_value = float(np.mean(list(models.values())))
        mos_pct = ((composite_fair_value - current_price) / composite_fair_value) * 100
    else:
        composite_fair_value = 0.0
        mos_pct = 0.0

    if composite_fair_value == 0:
        status = "Data Fundamental Tidak Mencukupi (EPS/BVPS Negatif)"
    elif mos_pct >= 15:
        status = "UNDERVALUED (Murah / Layak Koleksi) 🟢"
    elif mos_pct <= -15:
        status = "OVERVALUED (Mahal) 🔴"
    else:
        status = "FAIRLY VALUED (Harga Wajar) 🟡"

    return {
        "models": models,
        "composite": composite_fair_value,
        "mos_pct": mos_pct,
        "status": status,
    }
