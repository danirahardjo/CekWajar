import math
import numpy as np


def _detect_sector_profile(sector: str) -> tuple[str, dict]:
    """Menentukan profil bobot valuasi paling relevan sesuai sektor emiten."""
    sec = (sector or "").lower()

    if any(k in sec for k in ["financial", "bank", "insurance", "real estate", "property"]):
        return (
            "Keuangan & Properti (Fokus PBV & ROE)",
            {
                "pbv": 0.35,
                "justified_pbv": 0.35,
                "per": 0.20,
                "graham": 0.10,
                "peg": 0.00,
                "dcf": 0.00,
            },
        )

    if any(k in sec for k in ["energy", "basic materials", "mining", "utilities"]):
        return (
            "Komoditas & Siklikal (Fokus Aset & Siklus Laba)",
            {
                "pbv": 0.35,
                "per": 0.30,
                "graham": 0.20,
                "dcf": 0.15,
                "justified_pbv": 0.00,
                "peg": 0.00,
            },
        )

    return (
        "Pertumbuhan, Konsumsi & Umum (Fokus Laba & Arus Kas)",
        {
            "per": 0.30,
            "dcf": 0.25,
            "peg": 0.20,
            "pbv": 0.15,
            "graham": 0.10,
            "justified_pbv": 0.00,
        },
    )


def _compute_dcf_per_share(
    fcf_per_share: float,
    growth_rate: float,
    discount_rate: float,
    net_debt_per_share: float,
) -> float:
    """Menghitung nilai intrinsik 2-Stage DCF 5 tahun dengan terminal growth 4%."""
    if fcf_per_share <= 0:
        return 0.0

    terminal_g = 0.04
    r = max(discount_rate, terminal_g + 0.03)
    g_start = float(np.clip(growth_rate if growth_rate > 0 else 0.06, 0.04, 0.18))

    pv_fcf = 0.0
    projected_fcf = fcf_per_share
    for year in range(1, 6):
        # Laju pertumbuhan melandai secara bertahap menuju terminal growth
        g_year = g_start - ((g_start - terminal_g) * ((year - 1) / 5.0))
        projected_fcf *= 1.0 + g_year
        pv_fcf += projected_fcf / ((1.0 + r) ** year)

    terminal_val = (projected_fcf * (1.0 + terminal_g)) / (r - terminal_g)
    pv_terminal = terminal_val / ((1.0 + r) ** 5)

    enterprise_val_ps = pv_fcf + pv_terminal
    equity_val_ps = enterprise_val_ps - max(0.0, net_debt_per_share)
    return max(0.0, equity_val_ps)


def calculate_fair_value(
    current_price: float,
    eps: float,
    bvps: float,
    fair_per: float,
    fair_pbv: float,
    sector: str = "Tidak Tersedia",
    roe: float = 0.0,
    earnings_growth: float = 0.0,
    dividend_yield: float = 0.0,
    fcf_per_share: float = 0.0,
    net_debt_per_share: float = 0.0,
    discount_rate: float = 0.11,
) -> dict:
    """Menghitung harga wajar saham secara tertimbang (Sector-Aware Multi-Model Valuation)."""
    profile_name, base_weights = _detect_sector_profile(sector)

    raw_models = {}

    # 1. Benjamin Graham Number
    if eps > 0 and bvps > 0:
        raw_models["graham"] = ("Graham Number", math.sqrt(22.5 * eps * bvps))

    # 2. Relative PER Valuation
    if eps > 0:
        raw_models["per"] = (f"Valuasi PER ({fair_per:.1f}x)", eps * fair_per)

    # 3. Relative PBV Valuation
    if bvps > 0:
        raw_models["pbv"] = (f"Valuasi PBV ({fair_pbv:.1f}x)", bvps * fair_pbv)

    # 4. Justified PBV berbasis ROE: (ROE - g) / (r - g)
    if bvps > 0 and roe > 0.03:
        g_sustainable = float(np.clip(roe * 0.35, 0.03, 0.07))
        r_req = max(discount_rate, g_sustainable + 0.03)
        justified_mult = float(np.clip((roe - g_sustainable) / (r_req - g_sustainable), 0.4, 6.5))
        raw_models["justified_pbv"] = (
            f"Justified PBV (ROE {roe*100:.1f}% → {justified_mult:.2f}x)",
            bvps * justified_mult,
        )

    # 5. Peter Lynch Fair Value (PEG = 1.0: EPS * (Growth% + DivYield%))
    if eps > 0 and earnings_growth > 0.02:
        div_pct = (dividend_yield * 100) if dividend_yield < 1 else dividend_yield
        growth_pct = min(earnings_growth * 100, 25.0) + min(max(div_pct, 0.0), 8.0)
        lynch_mult = float(np.clip(growth_pct, 6.0, 25.0))
        raw_models["peg"] = (f"Peter Lynch Growth ({lynch_mult:.1f}x)", eps * lynch_mult)

    # 6. 2-Stage Discounted Cash Flow (DCF)
    if fcf_per_share > 0 and base_weights.get("dcf", 0) > 0:
        dcf_val = _compute_dcf_per_share(fcf_per_share, earnings_growth, discount_rate, net_debt_per_share)
        if dcf_val > 0:
            raw_models["dcf"] = ("DCF Arus Kas (5 Tahun)", dcf_val)

    # Outlier Guardrail: batasi nilai model yang terlalu ekstrem (>2.5x atau <0.35x median)
    models = {}
    details = []
    if raw_models:
        all_vals = [v[1] for v in raw_models.values()]
        med_val = float(np.median(all_vals))
        low_cap = med_val * 0.35
        high_cap = med_val * 2.50

        # Normalisasi bobot aktif agar total selalu 100%
        active_weights = {
            k: (base_weights.get(k, 0.15) if base_weights.get(k, 0.15) > 0 else 0.15)
            for k in raw_models
        }
        total_w = sum(active_weights.values())
        norm_weights = {k: w / total_w for k, w in active_weights.items()}

        composite_fair_value = 0.0
        for key, (label, val) in raw_models.items():
            guarded_val = float(np.clip(val, low_cap, high_cap))
            w = norm_weights[key]
            contrib = guarded_val * w
            composite_fair_value += contrib
            models[label] = guarded_val
            details.append({
                "Metode Valuasi": label,
                "Harga Wajar (Rp)": guarded_val,
                "Bobot (%)": w * 100,
                "Kontribusi (Rp)": contrib,
                "Potensi Upside/Downside (%)": ((guarded_val - current_price) / current_price) * 100,
            })

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
        "details": details,
        "profile": profile_name,
        "composite": composite_fair_value,
        "mos_pct": mos_pct,
        "status": status,
    }
