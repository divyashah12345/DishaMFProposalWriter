"""
Disha Wealth – Mutual Fund Investment Proposal Generator
=========================================================
Run:  streamlit run app.py
Deps: pip install streamlit pandas numpy requests reportlab openpyxl

Logo: Place your logo as  Dishaprintlogo.png  in the SAME folder as app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings
import io
import os
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, Image as RLImage,
                                 HRFlowable, PageBreak)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disha Wealth – MF Proposal", page_icon="🧭", layout="wide")

ADVISOR_NAME = "Divya Shah"
ADVISOR_ARN  = "ARN-339305"
RISK_FREE    = 0.065   # 6.5% p.a.
LOGO_PATH    = os.path.join(os.path.dirname(__file__), "Dishaprintlogo.png")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

NAVY  = colors.HexColor("#1B4F72")
LIGHT = colors.HexColor("#D6EAF8")
WHITE = colors.white
GOLD  = colors.HexColor("#F0A500")

# ─────────────────────────────────────────────────────────────
# OFFLINE FALLBACK
# ─────────────────────────────────────────────────────────────
OFFLINE_SAMPLE_FUNDS = {
    "HDFC Balanced Advantage Fund - Growth Plan":                  "100026",
    "ICICI Prudential Balanced Advantage Fund - Growth":           "120505",
    "ICICI Prudential Equity & Debt Fund - Growth":                "120586",
    "ICICI Prudential Multi-Asset Fund - Growth":                  "120600",
    "Edelweiss Gold and Silver ETF FOF - Regular Plan - Growth":   "145740",
    "Nippon India Multi Asset Allocation Fund - Regular Growth":   "148919",
    "HDFC Flexi Cap Fund - Growth Plan":                           "100033",
    "Franklin U.S. Opportunities Equity Active Fund of Fund - Regular Growth": "147622",
    "Bandhan Small Cap Fund - Regular Plan - Growth":              "147946",
    "Axis Greater China Equity Fund of Fund - Regular Growth":     "145169",
    "Nippon India Multi Cap Fund - Growth Plan - Growth Option":   "118701",
    "Nippon India Growth Fund - Regular Plan - Growth":            "118989",
    "Nippon India Small Cap Fund - Regular Plan - Growth":         "118778",
    "Nippon India Growth Mid Cap Fund - Growth Plan":              "118989",
    "Mirae Asset Large Cap Fund - Regular Growth":                 "118834",
    "ICICI Prudential Gilt Fund - Regular Growth":                 "120604",
    "Nippon India Gold Savings Fund - Regular Growth":             "118748",
}

DEFAULT_FUND_KEYWORDS = [
    "HDFC Balanced Advantage Fund",
    "ICICI Prudential Balanced Advantage Fund",
    "ICICI Prudential Equity & Debt Fund",
    "ICICI Prudential Multi-Asset Fund",
    "Edelweiss Gold and Silver ETF FOF",
    "Nippon India Multi Asset Allocation Fund",
    "HDFC Flexi Cap Fund",
    "Franklin U.S. Opportunities",
    "BANDHAN SMALL CAP FUND",
    "Bandhan Small Cap Fund",
    "Axis Greater China Equity Fund",
    "Nippon India Multi Cap Fund",
    "Nippon India Growth Fund",
    "Nippon India Small Cap Fund",
]


# ═══════════════════════════════════════════════════════════════
# MODULE 1 – AMFI FUND LIST
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_amfi_fund_list() -> dict:
    try:
        r = requests.get("https://api.mfapi.in/mf", headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {d["schemeName"]: str(d["schemeCode"]) for d in data}
    except Exception as e:
        st.warning(f"Could not fetch AMFI list ({e}). Using offline sample.")
        return OFFLINE_SAMPLE_FUNDS


# ═══════════════════════════════════════════════════════════════
# MODULE 2 – NAV + STATISTICS  (Max DD limited to last 3 years)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3_600, show_spinner=False)
def fetch_nav_stats(scheme_code: str) -> dict:
    empty = {
        "3Y CAGR": "-", "5Y CAGR": "-", "10Y CAGR": "-",
        "15Y CAGR": "-", "20Y CAGR": "-",
        "Std Dev": "-", "Sharpe": "-", "Sortino": "-",
        "Max DD (3Y)": "-",
        "_ret_list": None, "_ret_index": None,
    }
    if not scheme_code or scheme_code == "N/A":
        return empty

    try:
        r = requests.get(
            f"https://api.mfapi.in/mf/{scheme_code}",
            headers=HEADERS, timeout=25
        )
        r.raise_for_status()
        payload = r.json()

        df = pd.DataFrame(payload["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["nav"]).sort_values("date").set_index("date")

        if len(df) < 30:
            return empty

        df["ret"] = df["nav"].pct_change()
        df = df.dropna(subset=["ret"])

        latest_date = df.index[-1]
        latest_nav  = df["nav"].iloc[-1]

        result = dict(empty)

        for y, label in [(3,"3Y CAGR"),(5,"5Y CAGR"),(10,"10Y CAGR"),
                         (15,"15Y CAGR"),(20,"20Y CAGR")]:
            target = latest_date - pd.DateOffset(years=y)
            if df.index[0] <= target:
                idx      = df.index.get_indexer([target], method="nearest")[0]
                past_nav = df["nav"].iloc[idx]
                if past_nav > 0:
                    cagr = ((latest_nav / past_nav) ** (1.0 / y) - 1) * 100
                    result[label] = f"{cagr:.1f}%"

        ann_std = df["ret"].std() * np.sqrt(252)
        result["Std Dev"] = f"{ann_std * 100:.1f}%"

        ann_ret = (1 + df["ret"].mean()) ** 252 - 1

        if ann_std > 0:
            sharpe = (ann_ret - RISK_FREE) / ann_std
            result["Sharpe"] = f"{sharpe:.2f}"

        neg_rets = df["ret"][df["ret"] < 0]
        if len(neg_rets) > 5:
            down_std = neg_rets.std() * np.sqrt(252)
            if down_std > 0:
                sortino = (ann_ret - RISK_FREE) / down_std
                result["Sortino"] = f"{sortino:.2f}"

        cutoff_3y = latest_date - pd.DateOffset(years=3)
        df_3y = df[df.index >= cutoff_3y]
        if len(df_3y) >= 30:
            roll_max  = df_3y["nav"].cummax()
            drawdowns = (df_3y["nav"] / roll_max) - 1
            result["Max DD (3Y)"] = f"{drawdowns.min() * 100:.1f}%"
        else:
            roll_max  = df["nav"].cummax()
            drawdowns = (df["nav"] / roll_max) - 1
            result["Max DD (3Y)"] = f"{drawdowns.min() * 100:.1f}%*"

        result["_ret_list"]  = df["ret"].tolist()
        result["_ret_index"] = df.index.tolist()

        return result

    except Exception:
        return empty


def compute_beta_from_stats(fund_stats: dict, mkt_stats: dict) -> str:
    try:
        f_list = fund_stats.get("_ret_list")
        m_list = mkt_stats.get("_ret_list")
        f_idx  = fund_stats.get("_ret_index")
        m_idx  = mkt_stats.get("_ret_index")
        if not f_list or not m_list:
            return "-"
        f_series = pd.Series(f_list, index=f_idx)
        m_series = pd.Series(m_list, index=m_idx)
        aligned  = pd.concat([f_series, m_series], axis=1).dropna()
        if len(aligned) < 30:
            return "-"
        aligned.columns = ["f", "m"]
        cov  = np.cov(aligned["f"], aligned["m"])
        beta = cov[0][1] / cov[1][1]
        return f"{beta:.2f}"
    except Exception:
        return "-"


# ═══════════════════════════════════════════════════════════════
# MODULE 3 – SEBI CATEGORY MAP + ALLOCATION INFERENCE
# ═══════════════════════════════════════════════════════════════
FUND_CATEGORY_MAP = {
    "large cap":           ("Equity",             "Large Cap",          80, 10, 10),
    "index fund nifty":    ("Equity",             "Large Cap Index",    95,  3,  2),
    "index fund sensex":   ("Equity",             "Large Cap Index",    95,  3,  2),
    "nifty 50":            ("Equity",             "Large Cap Index",    95,  3,  2),
    "nifty next 50":       ("Equity",             "Large & Mid Cap",    60, 30, 10),
    "large & mid cap":     ("Equity",             "Large & Mid Cap",    50, 40, 10),
    "large and mid cap":   ("Equity",             "Large & Mid Cap",    50, 40, 10),
    "mid cap":             ("Equity",             "Mid Cap",            25, 65, 10),
    "small cap":           ("Equity",             "Small Cap",          15, 15, 70),
    "micro cap":           ("Equity",             "Small/Micro Cap",    10, 10, 80),
    "flexi cap":           ("Equity",             "Flexi Cap",          50, 30, 20),
    "multi cap":           ("Equity",             "Multi Cap",          35, 35, 30),
    "focused fund":        ("Equity",             "Focused",            55, 25, 20),
    "contra":              ("Equity",             "Contra/Value",       50, 30, 20),
    "value fund":          ("Equity",             "Value",              50, 30, 20),
    "dividend yield":      ("Equity",             "Dividend Yield",     60, 25, 15),
    "tax saver":           ("Equity/ELSS",        "ELSS",               50, 30, 20),
    "elss":                ("Equity/ELSS",        "ELSS",               50, 30, 20),
    "balanced advantage":  ("Hybrid",             "BAF",                55, 15, 10),
    "dynamic asset":       ("Hybrid",             "BAF",                55, 15, 10),
    "aggressive hybrid":   ("Hybrid",             "Aggressive Hybrid",  55, 25, 10),
    "equity & debt":       ("Hybrid",             "Aggressive Hybrid",  55, 25, 10),
    "equity and debt":     ("Hybrid",             "Aggressive Hybrid",  55, 25, 10),
    "conservative hybrid": ("Hybrid",             "Conservative Hybrid",15,  5,  5),
    "equity savings":      ("Hybrid",             "Equity Savings",     35, 10,  5),
    "multi asset":         ("Multi Asset",        "Multi Asset",        40, 15, 10),
    "asset allocation":    ("Multi Asset",        "Multi Asset",        40, 15, 10),
    "banking":             ("Sectoral",           "Banking",            90,  5,  5),
    "bank ":               ("Sectoral",           "Banking",            90,  5,  5),
    "financial services":  ("Sectoral",           "Financials",         85, 10,  5),
    "pharma":              ("Sectoral",           "Pharma",             75, 15, 10),
    "healthcare":          ("Sectoral",           "Healthcare",         70, 20, 10),
    "infra":               ("Sectoral",           "Infrastructure",     55, 25, 20),
    "infrastructure":      ("Sectoral",           "Infrastructure",     55, 25, 20),
    "technology":          ("Sectoral",           "Technology",         80, 12,  8),
    "it fund":             ("Sectoral",           "Technology",         80, 12,  8),
    "fmcg":                ("Sectoral",           "FMCG",               80, 12,  8),
    "consumption":         ("Sectoral",           "Consumption",        65, 20, 15),
    "manufacturing":       ("Sectoral",           "Manufacturing",      50, 28, 22),
    "psu equity":          ("Sectoral",           "PSU",                70, 20, 10),
    "energy":              ("Sectoral",           "Energy",             75, 15, 10),
    "defence":             ("Sectoral",           "Defence",            55, 28, 17),
    "real estate":         ("Sectoral",           "Real Estate",        75, 15, 10),
    "gilt":                ("Debt",               "Gilt",                0,  0,  0),
    "liquid":              ("Debt",               "Liquid",              0,  0,  0),
    "overnight":           ("Debt",               "Overnight",           0,  0,  0),
    "short duration":      ("Debt",               "Short Duration",      0,  0,  0),
    "medium duration":     ("Debt",               "Medium Duration",     0,  0,  0),
    "long duration":       ("Debt",               "Long Duration",       0,  0,  0),
    "corporate bond":      ("Debt",               "Corporate Bond",      0,  0,  0),
    "credit risk":         ("Debt",               "Credit Risk",         0,  0,  0),
    "money market":        ("Debt",               "Money Market",        0,  0,  0),
    "banking and psu":     ("Debt",               "Banking & PSU Debt",  0,  0,  0),
    "gold etf":            ("Gold/Commodity",     "Gold ETF",            0,  0,  0),
    "gold savings":        ("Gold/Commodity",     "Gold Fund",           0,  0,  0),
    "gold and silver":     ("Gold/Commodity",     "Gold & Silver",       0,  0,  0),
    "silver etf":          ("Gold/Commodity",     "Silver ETF",          0,  0,  0),
    "commodity":           ("Gold/Commodity",     "Commodity",           0,  0,  0),
    "nasdaq":              ("International",      "US Equity",           0,  0,  0),
    "s&p 500":             ("International",      "US Equity",           0,  0,  0),
    "us equity":           ("International",      "US Equity",           0,  0,  0),
    "u.s. opportunities":  ("International",      "US Equity",           0,  0,  0),
    "international":       ("International",      "International",       0,  0,  0),
    "global":              ("International",      "International",       0,  0,  0),
    "china":               ("International",      "China Equity",        0,  0,  0),
    "greater china":       ("International",      "China Equity",        0,  0,  0),
    "opportunities fund":  ("International/FOF",  "FOF",                 0,  0,  0),
    "fund of fund":        ("International/FOF",  "FOF",                 0,  0,  0),
}

_ASSET_EQUITY_PCT = {
    "Equity": 97, "Equity/ELSS": 97, "Sectoral": 97,
    "Hybrid": 70, "Multi Asset": 55,
    "Debt": 2,    "Gold/Commodity": 5, "International": 0, "International/FOF": 0,
}
_ASSET_DEBT_PCT = {
    "Equity": 0,  "Equity/ELSS": 0,  "Sectoral": 0,
    "Hybrid": 20, "Multi Asset": 25,
    "Debt": 95,   "Gold/Commodity": 0, "International": 0, "International/FOF": 0,
}
_ASSET_GOLD_PCT = {
    "Equity": 0,  "Equity/ELSS": 0,  "Sectoral": 0,
    "Hybrid": 0,  "Multi Asset": 15,
    "Debt": 0,    "Gold/Commodity": 92, "International": 0, "International/FOF": 0,
}
_ASSET_INTL_PCT = {
    "Equity": 0,  "Equity/ELSS": 0,  "Sectoral": 0,
    "Hybrid": 0,  "Multi Asset": 0,
    "Debt": 0,    "Gold/Commodity": 0, "International": 93, "International/FOF": 90,
}


def infer_allocation_from_name(scheme_name: str) -> dict:
    name_lower = scheme_name.lower()
    matched_key = None
    for kw in FUND_CATEGORY_MAP:
        if kw in name_lower:
            matched_key = kw
            break
    if matched_key:
        asset_class, category, lc, mc, sc = FUND_CATEGORY_MAP[matched_key]
    else:
        asset_class, category, lc, mc, sc = "Equity", "Unknown", 50, 30, 20

    eq   = _ASSET_EQUITY_PCT.get(asset_class, 95)
    debt = _ASSET_DEBT_PCT.get(asset_class, 0)
    gold = _ASSET_GOLD_PCT.get(asset_class, 0)
    intl = _ASSET_INTL_PCT.get(asset_class, 0)
    cash = max(0, 100 - eq - debt - gold - intl)

    if "balanced advantage" in name_lower or "dynamic asset" in name_lower:
        eq, debt, cash = 65, 25, 10

    return dict(asset_class=asset_class, category=category,
                large_cap=lc, mid_cap=mc, small_cap=sc,
                equity=eq, debt=debt, gold=gold, intl=intl, cash=cash)


@st.cache_data(ttl=3_600, show_spinner=False)
def fetch_portfolio_allocation_amfi(scheme_code: str, scheme_name: str) -> dict:
    try:
        url = (f"https://www.amfiindia.com/modules/PorfolioDisclousure"
               f"?loadPage=true&rn=1&sc={scheme_code}")
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and len(r.text) > 500:
            alloc = _parse_amfi_portfolio_html(r.text, scheme_name)
            if alloc:
                return alloc
    except Exception:
        pass

    try:
        r = requests.get(f"https://api.mfapi.in/mf/{scheme_code}",
                         headers=HEADERS, timeout=10)
        if r.status_code == 200:
            meta     = r.json().get("meta", {})
            combined = f"{scheme_name} {meta.get('scheme_category','')} {meta.get('scheme_type','')}".lower()
            base     = infer_allocation_from_name(scheme_name)
            if "large cap" in combined and "mid" not in combined:
                base.update(large_cap=80, mid_cap=10, small_cap=10)
            elif "mid cap" in combined:
                base.update(large_cap=25, mid_cap=65, small_cap=10)
            elif "small cap" in combined:
                base.update(large_cap=15, mid_cap=15, small_cap=70)
            base["source"] = "mfapi-meta"
            return base
    except Exception:
        pass

    base = infer_allocation_from_name(scheme_name)
    base["source"] = "Inferred (SEBI rules)"
    return base


def _parse_amfi_portfolio_html(html: str, scheme_name: str):
    try:
        eq_m   = re.search(r'Equity[^\d]*(\d+\.?\d*)\s*%', html, re.IGNORECASE)
        debt_m = re.search(r'Debt[^\d]*(\d+\.?\d*)\s*%',   html, re.IGNORECASE)
        gold_m = re.search(r'Gold[^\d]*(\d+\.?\d*)\s*%',   html, re.IGNORECASE)
        if eq_m or debt_m:
            eq   = float(eq_m.group(1))   if eq_m   else 0
            debt = float(debt_m.group(1)) if debt_m else 0
            gold = float(gold_m.group(1)) if gold_m else 0
            cash = max(0, 100 - eq - debt - gold)
            base = infer_allocation_from_name(scheme_name)
            ef   = eq / 100
            base.update(
                equity=round(eq,1), debt=round(debt,1),
                gold=round(gold,1), cash=round(cash,1),
                large_cap=round(base["large_cap"] * ef, 1),
                mid_cap  =round(base["mid_cap"]   * ef, 1),
                small_cap=round(base["small_cap"] * ef, 1),
                source="AMFI"
            )
            return base
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# MODULE 4 – PROJECTIONS
# ═══════════════════════════════════════════════════════════════
def sip_fv(monthly: float, rate_pa: float, years: int) -> float:
    r = rate_pa / 12
    n = years * 12
    if r == 0:
        return monthly * n
    return monthly * (((1 + r) ** n - 1) / r) * (1 + r)


def build_projection_table(monthly_sip, annual_topup, ret_pct, horizons=(3,5,8,10,15,20)):
    rows = []
    for y in horizons:
        fv = sip_fv(monthly_sip, ret_pct / 100, y)
        if annual_topup > 0:
            fv += sum(sip_fv(annual_topup / 12, ret_pct / 100, y - yi) for yi in range(y))
        invested = monthly_sip * y * 12 + annual_topup * y
        rows.append({
            "Year": y,
            "Total Invested":    f"Rs. {invested:,.0f}",
            "Probable Value":    f"Rs. {int(fv):,}",
            "Wealth Multiple":   f"{fv/invested:.1f}x" if invested else "-",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# MODULE 5 – WEIGHTED PORTFOLIO SUMMARY
# ═══════════════════════════════════════════════════════════════
def compute_weighted_allocation(fund_names, alloc_pcts):
    wt = dict(equity=0, debt=0, gold=0, intl=0, cash=0,
              large_cap=0, mid_cap=0, small_cap=0)
    for name, pct in zip(fund_names, alloc_pcts):
        w = pct / 100
        a = infer_allocation_from_name(name)
        for k in ["equity","debt","gold","intl","cash"]:
            wt[k] += a.get(k, 0) * w
        eq_w = a.get("equity", 0) / 100 * w
        wt["large_cap"] += a.get("large_cap", 0) * eq_w
        wt["mid_cap"]   += a.get("mid_cap",   0) * eq_w
        wt["small_cap"] += a.get("small_cap", 0) * eq_w
    return {k: round(v, 1) for k, v in wt.items()}


# ═══════════════════════════════════════════════════════════════
# MODULE 6 – PDF GENERATION  (landscape, word-wrap, proper cols)
# ═══════════════════════════════════════════════════════════════
PAGE_W, PAGE_H = landscape(A4)
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def _styles():
    base = getSampleStyleSheet()
    return {
        "title":    ParagraphStyle("T",  parent=base["Title"],   fontSize=16,
                                   textColor=NAVY, spaceAfter=4,  alignment=TA_CENTER),
        "h2":       ParagraphStyle("H2", parent=base["Heading2"],fontSize=9,
                                   textColor=NAVY, spaceBefore=10, spaceAfter=3),
        "normal":   base["Normal"],
        "small":    ParagraphStyle("SM", parent=base["Normal"],  fontSize=6.5,
                                   textColor=colors.grey),
        "cell":     ParagraphStyle("C",  parent=base["Normal"],  fontSize=6.5,
                                   wordWrap="CJK"),
        "cell_hdr": ParagraphStyle("CH", parent=base["Normal"],  fontSize=6.5,
                                   textColor=WHITE, fontName="Helvetica-Bold",
                                   wordWrap="CJK"),
        "meta":     ParagraphStyle("M",  parent=base["Normal"],  fontSize=7.5,
                                   textColor=colors.HexColor("#333333")),
    }


def _make_table(data_rows, headers, col_widths, styles_dict, font_size=6.5):
    """
    Build a ReportLab table with word-wrapped Paragraph cells.
    font_size: auto-reduced for wide tables (many columns) to prevent overflow.
    col_widths must always sum to exactly CONTENT_W.
    """
    ncols = len(headers)
    if ncols >= 11:
        font_size = min(font_size, 5.8)
    elif ncols >= 8:
        font_size = min(font_size, 6.2)

    base = getSampleStyleSheet()
    cs = ParagraphStyle("_c",  parent=base["Normal"], fontSize=font_size,
                        wordWrap="CJK", leading=font_size + 1.5)
    chs = ParagraphStyle("_ch", parent=base["Normal"], fontSize=font_size,
                         textColor=WHITE, fontName="Helvetica-Bold",
                         wordWrap="CJK", leading=font_size + 1.5)

    hdr = [Paragraph(str(h), chs) for h in headers]
    rows = [hdr]
    for row in data_rows:
        rows.append([Paragraph(str(v), cs) for v in row])

    pad = 2 if ncols >= 8 else 3

    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#AAAAAA")),
        ("TOPPADDING",    (0,0), (-1,-1), pad),
        ("BOTTOMPADDING", (0,0), (-1,-1), pad),
        ("LEFTPADDING",   (0,0), (-1,-1), 2),
        ("RIGHTPADDING",  (0,0), (-1,-1), 2),
    ]))
    return t


def generate_pdf(client_name, proj_df, comp_df, perf_df, alloc_df,
                 monthly_sip, annual_topup, risk_profile, wtd):

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=landscape(A4),
                              rightMargin=MARGIN, leftMargin=MARGIN,
                              topMargin=15*mm,   bottomMargin=15*mm)
    S    = _styles()
    story = []

    # ── HEADER ─────────────────────────────────────────────────
    logo_cell = ""
    if os.path.exists(LOGO_PATH):
        try:
            logo_cell = RLImage(LOGO_PATH, width=3.8*cm, height=1.3*cm)
        except Exception:
            logo_cell = Paragraph(f"<b>{ADVISOR_NAME}</b>", S["normal"])
    else:
        logo_cell = Paragraph(f"<b>Disha Wealth</b>", S["normal"])

    title_p = Paragraph(
        "<font color='#1B4F72'><b>Mutual Fund Investment Proposal</b></font>",
        ParagraphStyle("TP", fontSize=15, alignment=TA_RIGHT)
    )
    hdr_t = Table([[logo_cell, title_p]], colWidths=[6*cm, CONTENT_W - 6*cm])
    hdr_t.setStyle(TableStyle([
        ("VALIGN", (0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",  (1,0),(1,0),  "RIGHT"),
    ]))
    story.append(hdr_t)
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=5))

    # ── CLIENT META ────────────────────────────────────────────
    story.append(Paragraph(
        f"<b>Prepared for:</b> {client_name} &nbsp;&nbsp;&nbsp; "
        f"<b>Date:</b> {datetime.today().strftime('%d %b %Y')} &nbsp;&nbsp;&nbsp; "
        f"<b>Advisor:</b> {ADVISOR_NAME} | {ADVISOR_ARN} &nbsp;&nbsp;&nbsp; "
        f"<b>Risk Profile:</b> {risk_profile} &nbsp;&nbsp;&nbsp; "
        f"<b>Monthly SIP:</b> Rs.{monthly_sip:,.0f}",
        S["meta"]
    ))
    story.append(Spacer(1, 5))

    # ── WEIGHTED ALLOCATION SUMMARY BAR ───────────────────────
    story.append(Paragraph("<b>Portfolio Weighted Allocation (Estimated)</b>", S["h2"]))
    sum_data = [
        ["Equity %", "Debt %", "Gold %", "Intl %", "Cash %",
         "Large Cap", "Mid Cap", "Small Cap"],
        [f"{wtd['equity']}%", f"{wtd['debt']}%", f"{wtd['gold']}%",
         f"{wtd['intl']}%",   f"{wtd['cash']}%",
         f"{wtd['large_cap']}%", f"{wtd['mid_cap']}%", f"{wtd['small_cap']}%"],
    ]
    cw8 = [CONTENT_W / 8] * 8
    sum_t = Table(sum_data, colWidths=cw8)
    sum_t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",   (0,0),(-1,0), WHITE),
        ("BACKGROUND",  (0,1),(-1,1), LIGHT),
        ("ALIGN",       (0,0),(-1,-1),"CENTER"),
        ("FONTSIZE",    (0,0),(-1,-1), 7.5),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.grey),
        ("TOPPADDING",  (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    story.append(sum_t)
    story.append(Spacer(1, 5))

    # ── A: PROJECTION ──────────────────────────────────────────
    story.append(Paragraph("<b>A. Wealth Compounding Projection</b>", S["h2"]))
    _aw = [CONTENT_W * r for r in [0.08, 0.31, 0.33, 0.28]]
    story.append(_make_table(proj_df.values.tolist(), proj_df.columns.tolist(), _aw, S))
    story.append(Spacer(1, 6))

    # ── B: PORTFOLIO COMPOSITION ───────────────────────────────
    story.append(Paragraph("<b>B. Portfolio Composition</b>", S["h2"]))
    _b_scheme = CONTENT_W * 0.36
    _b_rest   = (CONTENT_W - _b_scheme) / max(1, len(comp_df.columns) - 1)
    bc = [_b_scheme] + [_b_rest] * (len(comp_df.columns) - 1)
    story.append(_make_table(comp_df.values.tolist(), comp_df.columns.tolist(), bc, S))
    story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ── C: PERFORMANCE & RISK ──────────────────────────────────
    story.append(Paragraph("<b>C. Scheme Performance &amp; Risk Metrics</b>", S["h2"]))
    _pc_scheme = CONTENT_W * 0.30          # Widened for full fund name visibility
    _pc_rest   = (CONTENT_W - _pc_scheme) / max(1, len(perf_df.columns) - 1)
    pc = [_pc_scheme] + [_pc_rest] * (len(perf_df.columns) - 1)
    story.append(_make_table(perf_df.values.tolist(), perf_df.columns.tolist(), pc, S))
    story.append(Paragraph(
        "Source: mfapi.in  |  Sharpe & Sortino: risk-free = 6.5% p.a.  |  "
        "Beta: vs Nifty 500 proxy  |  Max DD = 3-year rolling drawdown",
        S["small"]
    ))
    story.append(Spacer(1, 6))

    # ── D: ASSET & MARKET-CAP ALLOCATION ──────────────────────
    story.append(Paragraph("<b>D. Asset &amp; Market-Cap Allocation (per scheme)</b>", S["h2"]))
    _dc_scheme   = CONTENT_W * 0.26        # Widened for full fund name visibility
    _dc_category = CONTENT_W * 0.10
    _dc_rest     = (CONTENT_W - _dc_scheme - _dc_category) / max(1, len(alloc_df.columns) - 2)
    dc = [_dc_scheme, _dc_category] + [_dc_rest] * (len(alloc_df.columns) - 2)
    story.append(_make_table(alloc_df.values.tolist(), alloc_df.columns.tolist(), dc, S))
    story.append(Paragraph(
        "Primary source: AMFI → mfapi meta → SEBI category rule inference. "
        "Large/Mid/Small Cap % are of total portfolio, not just equity portion.",
        S["small"]
    ))
    story.append(Spacer(1, 6))

    # ── DISCLAIMER ─────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceBefore=8))
    story.append(Paragraph(
        "<b>Disclaimer:</b> Mutual fund investments are subject to market risks. "
        "Read all scheme-related documents carefully before investing. "
        "Projections are illustrative and assume constant returns; past performance is not a "
        "guarantee of future returns. Market-cap and allocation figures are estimated using "
        "SEBI category rules and may differ from actual holdings. "
        "This is not investment advice.",
        S["small"]
    ))

    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# MODULE 7 – CSV EXPORTS
# ═══════════════════════════════════════════════════════════════
def build_all_funds_csv(funds_dict: dict, mkt_stats: dict) -> bytes:
    rows = []
    for name, code in list(funds_dict.items()):
        a = infer_allocation_from_name(name)
        rows.append({
            "Scheme Name": name,
            "Scheme Code": code,
            "Asset Class": a["asset_class"],
            "Category":    a["category"],
            "Note": "Live CAGR not fetched for all funds (use selected funds CSV for metrics)",
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def build_selected_funds_csv(perf_df: pd.DataFrame, alloc_df: pd.DataFrame) -> bytes:
    merged = pd.merge(perf_df, alloc_df, on="Scheme", how="left")
    return merged.to_csv(index=False).encode("utf-8")


# ═══════════════════════════════════════════════════════════════
# STREAMLIT MAIN UI
# ═══════════════════════════════════════════════════════════════
def main():
    st.markdown(
        "<h1 style='text-align:center;color:#1B4F72'>🧭 Disha Wealth</h1>"
        "<h4 style='text-align:center;color:#555'>Your Compass to Financial Freedom</h4>",
        unsafe_allow_html=True
    )
    st.caption(f"Prepared By: **{ADVISOR_NAME}** | {ADVISOR_ARN}")

    if os.path.exists(LOGO_PATH):
        lcol, _, _ = st.columns([1,3,3])
        lcol.image(LOGO_PATH, width=140)
    else:
        st.info("📌 Place `Dishaprintlogo.png` in the same folder as `app.py` for logo in PDF.")
    st.divider()

    # ── Step 1: Client Details ──────────────────────────────
    st.subheader("📋 Step 1 – Client & Investment Details")
    c1, c2, c3, c4 = st.columns(4)
    client_name  = c1.text_input("Client Name", placeholder="e.g. Mukesh Badala")
    monthly_sip  = c2.number_input("Monthly SIP (Rs.)", value=10_000, step=5_000)
    annual_topup = c3.number_input("Annual Top-Up (Rs.)", value=1_000, step=1_000)
    horizon_yrs  = c4.number_input("Investment Horizon (yrs)", value=15, step=1,
                                   min_value=1, max_value=40)
    c5, c6 = st.columns(2)
    expected_ret = c5.number_input("Assumed Return (% p.a.)", value=12.0, step=0.5)
    risk_profile = c6.selectbox("Risk Profile",
                                ["Low","Moderate","Moderately High","High","Very High"])

    # ── Step 2: Fund Selection ──────────────────────────────
    st.divider()
    st.subheader("📦 Step 2 – Select Mutual Funds")

    with st.spinner("Loading AMFI fund list…"):
        funds_dict = fetch_amfi_fund_list()
    if not funds_dict:
        st.error("Could not load fund list."); return

    fund_names_all = sorted(funds_dict.keys())

    default_sel = []
    for kw in DEFAULT_FUND_KEYWORDS:
        for f in fund_names_all:
            if kw.lower() in f.lower():
                if f not in default_sel:
                    default_sel.append(f)
                break
    default_sel = default_sel[:13]

    selected_funds = st.multiselect(
        "Search & select funds (defaults = Disha recommended list):",
        fund_names_all, default=default_sel,
        help="Type fund name to search."
    )
    if not selected_funds:
        st.info("Select at least one fund to continue."); return

    # ── Step 3: Allocation ──────────────────────────────────
    st.divider()
    st.subheader("📊 Step 3 – Set SIP Allocation (%)")

    if "alloc_data" not in st.session_state:
        st.session_state.alloc_data = {}
    for f in selected_funds:
        if f not in st.session_state.alloc_data:
            equal = round(100 / len(selected_funds), 1)
            st.session_state.alloc_data[f] = equal
    for f in list(st.session_state.alloc_data):
        if f not in selected_funds:
            del st.session_state.alloc_data[f]

    cols = st.columns(min(len(selected_funds), 4))
    for i, fund in enumerate(selected_funds):
        st.session_state.alloc_data[fund] = cols[i % 4].number_input(
            f"{fund[:28]}…" if len(fund) > 90 else fund,
            value=float(st.session_state.alloc_data[fund]),
            min_value=0.0, max_value=100.0, step=0.5, key=f"alloc_{i}"
        )

    alloc_pcts  = st.session_state.alloc_data
    total_alloc = sum(alloc_pcts.values())
    st.metric("Total Allocation", f"{total_alloc:.1f}%",
              delta="✓ OK" if abs(total_alloc-100)<0.5 else f"{100-total_alloc:+.1f}% remaining")

    # ── Generate ────────────────────────────────────────────
    st.divider()
    go = st.button("🚀 Generate Proposal", type="primary", use_container_width=True)
    if not go:
        return
    if not client_name:
        st.error("Enter client name."); return
    if abs(total_alloc - 100) > 0.5:
        st.error("Allocations must sum to exactly 100%."); return

    # ════════════════════════════════════════════════════════
    # BUILD PROPOSAL DATA
    # ════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        f"## 📄 Investment Proposal — {client_name}\n"
        f"**Date:** {datetime.today().strftime('%d %b %Y')}  |  "
        f"**Advisor:** {ADVISOR_NAME} {ADVISOR_ARN}  |  "
        f"**Risk:** {risk_profile}  |  **Horizon:** {horizon_yrs} yrs"
    )

    # A: Projection
    st.subheader("📈 A. Wealth Compounding Projection")
    proj_df = build_projection_table(monthly_sip, annual_topup, expected_ret)
    st.dataframe(proj_df, use_container_width=True, hide_index=True)
    st.caption(f"Assumed {expected_ret}% p.a. | SIP Rs.{monthly_sip:,}/mo | Top-Up Rs.{annual_topup:,}/yr")

    # B: Composition
    st.subheader("📋 B. Portfolio Composition")
    comp_rows = []
    for fund, pct in alloc_pcts.items():
        a = infer_allocation_from_name(fund)
        comp_rows.append({
            "Scheme Name": fund,
            "Category":    a["category"],
            "Asset Class": a["asset_class"],
            "Alloc %":     f"{pct:.1f}%",
            "SIP (Rs.)":   f"Rs.{monthly_sip*pct/100:,.0f}",
            "Top-Up (Rs.)":f"Rs.{annual_topup*pct/100:,.0f}",
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # C: Performance & Risk  (Max DD = 3Y)
    st.subheader("📊 C. Scheme Performance & Risk Metrics")
    st.caption("ℹ️ Max Drawdown shown is for last 3 years only")
    with st.spinner("Fetching NAV data from mfapi.in…"):
        mkt_code  = funds_dict.get("Nippon India Nifty 500 Index Fund - Regular Growth",
                    funds_dict.get("Nippon India Multi Cap Fund - Regular Growth", "118701"))
        mkt_stats = fetch_nav_stats(mkt_code)

        perf_rows = []
        for fund, pct in alloc_pcts.items():
            code  = funds_dict.get(fund, "")
            stats = fetch_nav_stats(code) if code else {}
            beta  = compute_beta_from_stats(stats, mkt_stats)
            # Fund names are fully preserved (no truncation)
            perf_rows.append({
                "Scheme":       fund,
                "Alloc %":      f"{pct:.1f}%",
                "3Y CAGR":      stats.get("3Y CAGR",  "-"),
                "5Y CAGR":      stats.get("5Y CAGR",  "-"),
                "10Y CAGR":     stats.get("10Y CAGR", "-"),
                "15Y CAGR":     stats.get("15Y CAGR", "-"),
                "Std Dev":      stats.get("Std Dev",  "-"),
                "Sharpe":       stats.get("Sharpe",   "-"),
                "Sortino":      stats.get("Sortino",  "-"),
                "Max DD (3Y)":  stats.get("Max DD (3Y)", "-"),
                "Beta":         beta,
            })

    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # D: Asset & MCap Allocation
    st.subheader("🏗️ D. Asset & Market-Cap Allocation")
    with st.spinner("Fetching portfolio allocation data…"):
        alloc_rows = []
        for fund, pct in alloc_pcts.items():
            code = funds_dict.get(fund, "")
            a    = fetch_portfolio_allocation_amfi(code, fund)
            # Fund names are fully preserved (no truncation)
            alloc_rows.append({
                "Scheme":      fund,
                "Category":    a["category"],
                "Alloc %":     f"{pct:.1f}%",
                "Equity %":    f"{a.get('equity','-')}",
                "Debt %":      f"{a.get('debt','-')}",
                "Gold %":      f"{a.get('gold','-')}",
                "Intl %":      f"{a.get('intl','-')}",
                "Cash %":      f"{a.get('cash','-')}",
                "Large Cap %": f"{a.get('large_cap','-')}",
                "Mid Cap %":   f"{a.get('mid_cap','-')}",
                "Small Cap %": f"{a.get('small_cap','-')}",
                "Source":      a.get("source", "Inferred"),
            })
    alloc_df = pd.DataFrame(alloc_rows)
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)
    st.caption("Primary source: AMFI → mfapi meta → SEBI category rule inference")

    # Weighted Average
    funds_list = list(alloc_pcts.keys())
    pcts_list  = list(alloc_pcts.values())
    wtd        = compute_weighted_allocation(funds_list, pcts_list)
    st.markdown("**📌 Portfolio Weighted Average Allocation**")
    col_list = st.columns(8)
    for col, (label, key) in zip(col_list, [
        ("Equity","equity"),("Debt","debt"),("Gold","gold"),("Intl","intl"),("Cash","cash"),
        ("Large Cap","large_cap"),("Mid Cap","mid_cap"),("Small Cap","small_cap")
    ]):
        col.metric(label, f"{wtd[key]}%")

    with st.expander("📜 Disclaimer"):
        st.markdown(
            "This proposal is prepared by an AMFI-registered Mutual Fund Distributor. "
            "Mutual fund investments are subject to market risks. Read all scheme-related "
            "documents carefully. Past performance is not a guarantee of future returns. "
            "Projections are illustrative only."
        )

    # ── EXPORTS ────────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Export")

    ec1, ec2 = st.columns(2)

    # ── CSV 1: All AMFI funds list ─────────────────────────────
    all_csv = build_all_funds_csv(funds_dict, mkt_stats)
    ec1.download_button(
        "📥 All AMFI Funds (CSV)",
        data=all_csv,
        file_name=f"AMFI_All_Funds_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ── CSV 2: Selected funds with performance + allocation ────
    selected_csv = build_selected_funds_csv(perf_df, alloc_df)
    ec2.download_button(
        "📥 Selected Funds Metrics (CSV)",
        data=selected_csv,
        file_name=f"Selected_Funds_{client_name.replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ── Excel ──────────────────────────────────────────────────
    xls_buf = io.BytesIO()
    with pd.ExcelWriter(xls_buf, engine="openpyxl") as w:
        proj_df.to_excel(w,     sheet_name="Projection",   index=False)
        comp_df.to_excel(w,     sheet_name="Portfolio",    index=False)
        perf_df.to_excel(w,     sheet_name="Performance",  index=False)
        alloc_df.to_excel(w,    sheet_name="Allocation",   index=False)

    # ── PDF (landscape, word-wrapped) ─────────────────────────
    with st.spinner("Generating PDF…"):
        pdf_bytes = generate_pdf(
            client_name, proj_df, comp_df, perf_df, alloc_df,
            monthly_sip, annual_topup, risk_profile, wtd
        )

    col_xls, col_pdf = st.columns(2)
    col_xls.download_button(
        "📊 Download Excel",
        data=xls_buf.getvalue(),
        file_name=f"MF_Proposal_{client_name.replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    col_pdf.download_button(
        "📄 Download PDF (Landscape)",
        data=pdf_bytes,
        file_name=f"MF_Proposal_{client_name.replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )


if __name__ == "__main__":
    main()