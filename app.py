"""
EGN 330: Coastal Wave Mechanics — Master Computational Engine (v5.0)
Fully Merged Implementation: Modules M07 through M16 + Open-Ended Exam Q&A Synthesizer (Exam-Compliant High-Contrast GoodNotes UI)
Dr. Zhangping (John) Wei, UNCW
"""
import math
import streamlit as st
import numpy as np
from scipy.optimize import root_scalar

# ==========================================
# 1. PAGE CONFIGURATION & HIGH-CONTRAST STYLING
# ==========================================
st.set_page_config(
    page_title="EGN 330: Master Engine v5.0 (M07–M16 + Q&A)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #050507; color: #f8fafc; }
    .main-title { font-size: 1.85rem; font-weight: 800; color: #facc15; margin-bottom: 0.15rem; letter-spacing: -0.02em; }
    .sub-title { font-size: 0.95rem; color: #cbd5e1; margin-bottom: 1.0rem; font-weight: 500; }
    
    /* GoodNotes Top Knowns Box with Red-Derived Variable Symbols */
    .gn-knowns-box {
        background-color: #09090b;
        border: 2px solid #52525b;
        border-radius: 8px;
        padding: 12px 14px;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        line-height: 1.8;
    }
    .known-badge {
        display: inline-block;
        background-color: #14080c;
        border: 1.5px solid #f43f5e;
        border-radius: 6px;
        padding: 2px 9px;
        margin: 3px 6px 3px 0;
    }
    .known-var {
        color: #ff4d6d;
        font-weight: 800;
        text-shadow: 0 0 8px rgba(255, 77, 109, 0.35);
    }
    .known-val {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* High-Contrast Bright Neon Green Find Box */
    .gn-find-box {
        background: linear-gradient(180deg, #021a0e 0%, #010f08 100%);
        border: 3px solid #22c55e;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'Courier New', monospace;
        font-size: 1.14rem;
        font-weight: 800;
        color: #4ade80;
        margin-bottom: 14px;
        box-shadow: 0 0 18px rgba(34, 197, 94, 0.25);
    }
    
    /* GoodNotes Formula Strip */
    .gn-formula-item {
        background-color: #090d16;
        border: 1.5px solid #6366f1;
        border-left: 5px solid #818cf8;
        border-radius: 6px;
        padding: 8px 12px;
        font-family: 'Courier New', monospace;
        font-size: 1.0rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    
    /* High-Contrast Amber Oval Pill for Intermediate Calculations */
    .gn-oval-pill {
        background: linear-gradient(180deg, #271705 0%, #170d02 100%);
        border: 2.5px solid #f59e0b;
        border-radius: 9999px;
        padding: 10px 18px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
    }
    .gn-oval-label {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #fbbf24;
        letter-spacing: 0.04em;
    }
    .gn-oval-val {
        font-family: 'Courier New', monospace;
        font-size: 1.28rem;
        font-weight: 800;
        color: #fef08a;
    }
    
    /* High-Contrast Emerald Rectangular Box for Final Answers */
    .gn-final-box {
        background: linear-gradient(180deg, #042f2e 0%, #021a19 100%);
        border: 3px solid #10b981;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.22);
    }
    .gn-final-label {
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #6ee7b7;
        letter-spacing: 0.04em;
    }
    .gn-final-val {
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 4px 0;
    }
    .gn-final-note {
        font-size: 0.82rem;
        color: #a7f3d0;
        font-weight: 600;
    }
    
    /* Full-Width GoodNotes Bottom Grey Explanation Box */
    .gn-grey-box {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 3px solid #cbd5e1;
        border-radius: 10px;
        padding: 18px 22px;
        color: #ffffff;
        font-family: 'Courier New', monospace;
        font-size: 1.12rem;
        line-height: 1.6;
        font-weight: 700;
        margin-top: 10px;
        width: 100%;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.5);
    }
    
    /* Exam Trap Radar Alert */
    .trap-alert {
        background: linear-gradient(90deg, #3b0712 0%, #27050c 100%);
        border: 2px solid #fb7185;
        border-left: 6px solid #f43f5e;
        border-radius: 8px;
        padding: 10px 14px;
        color: #ffe4e6;
        font-size: 0.92rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. WIEGEL (1962) DIFFRACTION TABLE & CORE MATH
# ==========================================
WIEGEL_BETAS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
WIEGEL_DATA = {
    15: {0.5:[0.49,0.79,0.83,0.9,0.97,1.01,1.03,1.02,1.01,0.99,0.99,1,1], 1:[0.38,0.73,0.83,0.95,1.04,1.04,0.99,0.98,1.01,1.01,1,1,1], 2:[0.21,0.68,0.86,1.05,1.03,0.97,1.02,0.99,1,1,1,1,1], 5:[0.13,0.63,0.99,1.04,1.03,1.02,0.99,0.99,1,1.01,1,1,1], 10:[0.35,0.58,1.1,1.05,0.98,0.99,1.01,1,1,1,1,1,1]},
    30: {0.5:[0.61,0.63,0.68,0.76,0.87,0.97,1.03,1.05,1.03,1.01,0.99,0.95,1], 1:[0.5,0.53,0.63,0.78,0.95,1.06,1.05,0.98,0.98,1.01,1.01,0.97,1], 2:[0.4,0.44,0.59,0.84,1.07,1.03,0.96,1.02,0.98,1.01,0.99,0.95,1], 5:[0.27,0.32,0.55,1,1.04,1.04,1.02,0.99,0.99,1,1.01,0.97,1], 10:[0.2,0.24,0.54,1.12,1.06,0.97,0.99,1.01,1,1,1,0.98,1]},
    45: {0.5:[0.49,0.5,0.55,0.63,0.73,0.85,0.96,1.04,1.06,1.04,1,0.99,1], 1:[0.38,0.4,0.47,0.59,0.76,0.95,1.07,1.06,0.98,0.97,1.01,1.01,1], 2:[0.29,0.31,0.39,0.56,0.83,1.08,1.04,0.96,1.03,0.98,1.01,1,1], 5:[0.18,0.2,0.29,0.54,1.01,1.04,1.05,1.03,1,0.99,1.01,1,1], 10:[0.13,0.15,0.22,0.53,1.13,1.07,0.96,0.98,1.02,0.99,1,1,1]},
    60: {0.5:[0.4,0.41,0.45,0.52,0.6,0.72,0.85,1.13,1.04,1.06,1.03,1.01,1], 1:[0.31,0.32,0.36,0.44,0.57,0.75,0.96,1.08,1.06,0.98,0.98,1.01,1], 2:[0.22,0.23,0.28,0.37,0.55,0.83,1.08,1.04,0.96,1.03,0.98,1.01,1], 5:[0.14,0.15,0.18,0.28,0.53,1.01,1.04,1.05,1.03,0.99,0.99,1,1], 10:[0.1,0.11,0.13,0.21,0.52,1.14,1.07,0.96,0.98,1.01,1,1,1]},
    75: {0.5:[0.34,0.35,0.38,0.42,0.5,0.59,0.71,0.85,0.97,1.04,1.05,1.02,1], 1:[0.25,0.26,0.29,0.34,0.43,0.56,0.75,0.95,1.02,1.06,0.98,0.98,1], 2:[0.18,0.19,0.22,0.26,0.36,0.54,0.83,1.09,1.04,0.96,1.03,0.99,1], 5:[0.12,0.12,0.13,0.17,0.27,0.52,1.01,1.04,1.05,1.03,0.99,0.99,1], 10:[0.08,0.08,0.1,0.13,0.2,0.52,1.14,1.07,0.96,0.98,1.01,1,1]},
    90: {0.5:[0.31,0.31,0.33,0.36,0.41,0.49,0.59,0.71,0.85,0.96,1.03,1.03,1], 1:[0.22,0.23,0.24,0.28,0.33,0.42,0.56,0.75,0.96,1.07,1.05,0.99,1], 2:[0.16,0.16,0.18,0.2,0.26,0.35,0.54,0.69,1.08,1.04,0.96,1.02,1], 5:[0.1,0.1,0.11,0.13,0.16,0.27,0.53,1.01,1.04,1.05,1.02,0.99,1], 10:[0.07,0.07,0.08,0.09,0.13,0.2,0.52,1.14,1.07,0.96,0.99,1.01,1]},
    105: {0.5:[0.28,0.28,0.29,0.32,0.35,0.41,0.49,0.59,0.72,0.85,0.97,1.01,1], 1:[0.2,0.2,0.24,0.23,0.27,0.33,0.42,0.56,0.75,0.95,1.06,1.04,1], 2:[0.14,0.14,0.13,0.17,0.2,0.25,0.35,0.54,0.83,1.08,1.03,0.97,1], 5:[0.09,0.09,0.1,0.11,0.13,0.17,0.27,0.52,1.02,1.04,1.04,1.02,1], 10:[0.07,0.06,0.08,0.08,0.09,0.12,0.2,0.52,1.14,1.07,0.97,0.99,1]},
    120: {0.5:[0.25,0.26,0.27,0.28,0.31,0.35,0.41,0.5,0.6,0.73,0.87,0.97,1], 1:[0.18,0.19,0.19,0.21,0.23,0.27,0.33,0.43,0.57,0.76,0.95,1.04,1], 2:[0.13,0.13,0.14,0.14,0.17,0.2,0.26,0.16,0.55,0.83,1.07,1.03,1], 5:[0.08,0.08,0.08,0.09,0.11,0.13,0.16,0.27,0.53,1.01,1.04,1.03,1], 10:[0.06,0.06,0.06,0.07,0.07,0.09,0.13,0.2,0.52,1.13,1.06,0.98,1]},
    135: {0.5:[0.24,0.24,0.25,0.26,0.28,0.32,0.36,0.42,0.52,0.63,0.76,0.9,1], 1:[0.18,0.17,0.18,0.19,0.21,0.23,0.28,0.34,0.44,0.59,0.78,0.95,1], 2:[0.12,0.12,0.13,0.14,0.14,0.17,0.2,0.26,0.37,0.56,0.84,1.05,1], 5:[0.08,0.07,0.08,0.08,0.09,0.11,0.13,0.17,0.28,0.54,1,1.04,1], 10:[0.05,0.06,0.06,0.06,0.07,0.08,0.09,0.13,0.21,0.53,1.12,1.05,1]},
    150: {0.5:[0.23,0.23,0.24,0.25,0.27,0.29,0.33,0.38,0.45,0.55,0.68,0.83,1], 1:[0.16,0.17,0.17,0.18,0.19,0.22,0.24,0.29,0.36,0.47,0.63,0.83,1], 2:[0.12,0.12,0.12,0.13,0.14,0.15,0.18,0.22,0.28,0.39,0.59,0.86,1], 5:[0.07,0.07,0.08,0.08,0.08,0.1,0.11,0.13,0.18,0.29,0.55,0.99,1], 10:[0.05,0.05,0.05,0.06,0.06,0.07,0.08,0.1,0.13,0.22,0.54,1.1,1]},
    165: {0.5:[0.23,0.23,0.23,0.24,0.26,0.28,0.31,0.35,0.41,0.5,0.63,0.79,1], 1:[0.16,0.16,0.17,0.17,0.19,0.2,0.23,0.26,0.32,0.4,0.53,0.73,1], 2:[0.11,0.11,0.12,0.12,0.13,0.14,0.16,0.19,0.23,0.31,0.44,0.68,1], 5:[0.07,0.07,0.07,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.32,0.63,1], 10:[0.05,0.05,0.05,0.06,0.06,0.06,0.07,0.08,0.11,0.11,0.21,0.58,1]},
    180: {0.5:[0.2,0.25,0.23,0.24,0.25,0.28,0.31,0.34,0.4,0.49,0.61,0.78,1], 1:[0.1,0.17,0.16,0.18,0.18,0.23,0.22,0.25,0.31,0.38,0.5,0.7,1], 2:[0.02,0.09,0.12,0.12,0.13,0.18,0.16,0.18,0.22,0.29,0.4,0.6,1], 5:[0.02,0.06,0.07,0.07,0.07,0.08,0.1,0.12,0.14,0.18,0.27,0.46,1], 10:[0.01,0.05,0.05,0.06,0.06,0.07,0.07,0.08,0.1,0.13,0.2,0.36,1]}
}

def lookup_wiegel_kd(theta_deg: float, r_over_L: float, beta_deg: float):
    thetas = sorted(WIEGEL_DATA.keys())
    r_ls = [0.5, 1, 2, 5, 10]
    near_theta = min(thetas, key=lambda t: abs(t - theta_deg))
    near_rl = min(r_ls, key=lambda rl: abs(rl - r_over_L))
    near_beta_idx = min(range(len(WIEGEL_BETAS)), key=lambda i: abs(WIEGEL_BETAS[i] - beta_deg))
    return WIEGEL_DATA[near_theta][near_rl][near_beta_idx], near_theta, near_rl, WIEGEL_BETAS[near_beta_idx]

@st.cache_data
def solve_dispersion_relation(h: float, T: float, g: float = 9.81, method: str = "Beji (2013) [Dr. Wei Key]"):
    sigma = 2.0 * np.pi / T
    L0 = 1.56 * (T ** 2) if "1.56" in method else g * (T ** 2) / (2.0 * np.pi)
    C0 = L0 / T
    Cg0 = 0.5 * C0
    if h is None or h <= 0 or (h / L0) > 10.0:
        k0 = 2.0 * np.pi / L0
        return {"L0": L0, "C0": C0, "Cg0": Cg0, "L": L0, "C": C0, "k": k0, "kh": np.inf, "n": 0.5, "Cg": Cg0, "sigma": sigma}

    if "Beji" in method:
        kh0 = (2.0 * np.pi * h) / L0
        kh = kh0 * (1.0 + (kh0 ** 1.09) * np.exp(-(1.55 + 1.3 * kh0 + 0.216 * (kh0 ** 2)))) / np.sqrt(np.tanh(kh0))
        L = (2.0 * np.pi * h) / kh
        k = 2.0 * np.pi / L
    else:
        k0 = 2.0 * np.pi / L0
        y = k0 * h
        x = y * (1.0 + (y ** 1.09) * np.exp(-(1.55 + 1.3 * y + 0.216 * (y ** 2)))) ** (-0.5)
        k_seed = x / h
        def disp_func(k_val): return sigma ** 2 - g * k_val * np.tanh(k_val * h)
        def disp_fprime(k_val): return -g * (np.tanh(k_val * h) + k_val * h * (1.0 - np.tanh(k_val * h) ** 2))
        try:
            sol = root_scalar(disp_func, fprime=disp_fprime, x0=k_seed, method='newton', rtol=1e-12)
            k = sol.root
        except Exception:
            sol = root_scalar(disp_func, bracket=[0.0001, 10.0], method='brentq')
            k = sol.root
        L = 2.0 * np.pi / k
        kh = k * h

    C = L / T
    if kh > 15.0: n = 0.5
    elif kh < 0.001: n = 1.0
    else: n = 0.5 * (1.0 + (2.0 * kh) / np.sinh(2.0 * kh))
    Cg = n * C
    return {"L0": L0, "C0": C0, "Cg0": Cg0, "L": L, "C": C, "k": k, "kh": kh, "n": n, "Cg": Cg, "sigma": sigma}

def format_knowns_html(knowns_str: str) -> str:
    import re
    segments = [s.strip() for s in re.split(r'\s{2,}', knowns_str) if s.strip()]
    out = []
    for seg in segments:
        if "=" in seg:
            lhs, rhs = seg.split("=", 1)
            out.append(f'<span class="known-badge"><span class="known-var">{lhs.strip()}</span> = <span class="known-val">{rhs.strip()}</span></span>')
        else:
            out.append(f'<span class="known-badge"><span class="known-val">{seg}</span></span>')
    return "".join(out)

def render_goodnotes_terminal(knowns_str: str, find_str: str, formulas: list, ovals: list, finals: list, explanation: str, traps: list = None):
    if traps:
        for title, body in traps:
            st.markdown(f'<div class="trap-alert"><strong>⚠️ {title}:</strong> {body}</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([5, 7])
    with col_left:
        st.markdown("#### 1. GoodNotes Top Line & Bright Green 'Find' Box")
        st.markdown(f'<div class="gn-knowns-box">{format_knowns_html(knowns_str)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gn-find-box"><span style="color:#86efac;text-transform:uppercase;">🎯 Find:</span> {find_str}</div>', unsafe_allow_html=True)
        st.markdown("#### 2. Governing Equations to Write")
        for eq in formulas:
            st.markdown(f'<div class="gn-formula-item">{eq}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 3. Oval Intermediate Calculations (Circle These in GoodNotes)")
        oval_cols = st.columns(3)
        for idx, (label, sym, val, unit) in enumerate(ovals):
            with oval_cols[idx % 3]:
                st.markdown(
                    f'<div class="gn-oval-pill"><div class="gn-oval-label">{label}</div>'
                    f'<div class="gn-oval-val">{sym} = {val} <span style="font-size:0.9rem;color:#fbbf24;">{unit}</span></div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### 4. Rectangular Boxed Final Answers (Box These in GoodNotes)")
        fin_cols = st.columns(2)
        for idx, (label, sym, val, unit, note) in enumerate(finals):
            with fin_cols[idx % 2]:
                st.markdown(
                    f'<div class="gn-final-box"><div class="gn-final-label">{label}</div>'
                    f'<div class="gn-final-val">{sym} = {val} <span style="font-size:1.05rem;color:#6ee7b7;">{unit}</span></div>'
                    f'<div class="gn-final-note">{note}</div></div>',
                    unsafe_allow_html=True,
                )

    # FULL-WIDTH SECTION 5 (STARTS ALL THE WAY ON THE LEFT & TAKES UP THE WHOLE PAGE)
    st.markdown("#### 5. GoodNotes Bottom Grey Explanation Box (Full-Page Width)")
    st.markdown(f'<div class="gn-grey-box">{explanation}</div>', unsafe_allow_html=True)
    st.code(explanation, language=None)

# ==========================================
# 3. SIDEBAR NAVIGATION & GLOBAL CONSTANTS
# ==========================================
with st.sidebar:
    st.title("🌊 EGN 330 Engine v5.0")
    selected_module = st.radio(
        "Select Exam Module:",
        (
            "🔥 M15: Nonlinear Wave Theories",
            "🔥 M16: Solitary Waves & Tsunamis",
            "🧠 Open-Ended Exam Q&A Synthesizer",
            "🔥 M14: Wave Set-Up & Set-Down",
            "🔥 M12: Wave Breaking & Run-Up",
            "🔥 M13: Wave Radiation Stress",
            "M07: Wave Energy & Power",
            "M08: 1D Wave Shoaling",
            "M09: Shoaling & Refraction",
            "M10: Standing Waves & Pressure",
            "M11: Breakwater Diffraction",
        ),
    )
    st.markdown("---")
    st.subheader("Global Constants & Solver")
    rho_in = st.number_input("Water Density ρ [kg/m³]", value=1000.0, step=5.0, help="1000 = Dr. Wei Practice Default, 1025 = Seawater")
    g_in = st.number_input("Gravity g [m/s²]", value=9.81, step=0.01)
    solver_method = st.selectbox("Dispersion Solver:", ("Beji (2013) [Dr. Wei Key]", "Exact Newton-Raphson", "Handwritten (L0 = 1.56*T^2)"))

# ==========================================
# 3B. LE MÉHAUTÉ CLASSIFIER & OPEN-ENDED Q&A BANK
# ==========================================
def classify_le_mehaute(h: float, T: float, H: float, g: float, L: float, L0: float):
    dim_H = H / (g * (T ** 2))
    dim_h = h / (g * (T ** 2))
    Ur = (H * (L ** 2)) / (h ** 3)
    H_over_h = H / h
    H_over_L0 = H / L0

    if H_over_h >= 0.78 or H_over_L0 >= 0.142:
        return (
            "Breaking Wave Regime (Exceeds Le Méhauté Stability Limit)",
            f"Wave exceeds breaking stability limit (H/h = {H_over_h:.3f}, H/L₀ = {H_over_L0:.3f}).",
        )
    if Ur >= 26.0 or dim_h < 0.005:
        if H_over_h > 0.4:
            return (
                "Solitary Wave Theory / 5th-Order Stream Function Theory",
                f"Large Ursell number (U_r = {Ur:.2f} ≥ 26) indicates strong shallow-water nonlinearity.",
            )
        return (
            "Cnoidal Wave Theory",
            f"Large Ursell number (U_r = {Ur:.2f} ≥ 26) indicates strong shallow-water nonlinearity, consistent with selecting Cnoidal wave theory.",
        )
    if dim_H <= 0.0008 and Ur < 5.0:
        return (
            "Linear (Small-Amplitude) Wave Theory",
            f"Very small Ursell number (U_r = {Ur:.2f}) and small dimensionless wave height (H/gT² = {dim_H:.5f}) indicate negligible nonlinearity, consistent with Linear Wave Theory.",
        )
    if dim_h >= 0.02:
        if dim_H <= 0.0055:
            return (
                "Stokes 2nd-Order Wave Theory",
                f"The Ursell number (U_r = {Ur:.2f}) is small, indicating weak shallow-water nonlinearity. It is consistent with selecting a Stokes-type theory.",
            )
        elif dim_H <= 0.011:
            return (
                "Stokes 3rd-Order Wave Theory",
                f"The Ursell number (U_r = {Ur:.2f}) is small while wave steepness is moderate-to-high, consistent with a higher-order Stokes wave theory.",
            )
        else:
            return (
                "Stokes 4th-Order / 5th-Order Stream Function Theory",
                f"High dimensionless wave height (H/gT² = {dim_H:.4f}) requires Stokes 4th-order or 5th-order Stream Function theory.",
            )
    else:
        if Ur >= 15.0:
            return (
                "Stokes 3rd/4th-Order or Stream Function (Near Cnoidal Boundary U_r ≈ 26)",
                f"The Ursell number (U_r = {Ur:.2f}) approaches the transition range (U_r ≈ 20–26) between Stokes-type and Cnoidal-type waves.",
            )
        return (
            "Stokes 2nd-Order Wave Theory",
            f"The Ursell number (U_r = {Ur:.2f}) is below the 20–26 transition, consistent with selecting a Stokes-type wave theory.",
        )


EGN330_QA_BANK = [
    {
        "mod": "M15",
        "title": "Limitations of Linear Wave Theory & Linearization Assumptions (M15 Slides 4–7)",
        "keywords": "limitations linear wave theory small amplitude linearization free surface boundary condition kfsbc dfsbc z=0 h/l h/h laplace",
        "question": "What are the limitations of linear (small-amplitude) wave theory, and how was linearization carried out?",
        "answer": "Linear (small-amplitude) wave theory solves the Laplace equation (∇²φ = 0) across the full range of relative water depth (h/L), but it is strictly limited to waves with very small amplitude relative to both water depth (H/h ≪ 1 for shallow water) and wavelength (H/L ≪ 1 for deep water). During linearization, nonlinear velocity-squared terms ½[(∂φ/∂x)² + (∂φ/∂z)²] and product terms (∂φ/∂x)(∂η/∂x) are dropped, and the Kinematic and Dynamic Free Surface Boundary Conditions (KFSBC and DFSBC) are applied at the still water level (z = 0) rather than at the actual moving water surface (z = η).",
        "equations": "∇²φ = 0,   H/h ≪ 1,   H/L ≪ 1,   BCs applied at z = 0 instead of z = η(x,t)",
        "trap": "Linear theory works across ALL relative depths (h/L from deep to shallow), as long as wave height H is small compared to h and L!",
    },
    {
        "mod": "M15",
        "title": "When Do Nonlinear Wave Effects Matter? (M15 Slide 9)",
        "keywords": "when nonlinear effects matter steepness 0.05 relative depth 0.5 bathymetry wave forces structures",
        "question": "When do nonlinear wave effects matter in coastal engineering?",
        "answer": "Nonlinear wave effects matter when: (1) Wave steepness (H/L) exceeds approximately 0.05; (2) Relative water depth (h/L) is less than 0.5 (intermediate to shallow water); (3) Waves propagate over varying bathymetry; (4) Wave amplitudes are large relative to local water depth (finite H/h); and (5) Accurate prediction of wave forces on coastal or offshore structures is required.",
        "equations": "H/L > 0.05,   h/L < 0.5,   Finite H/h",
        "trap": "List at least 4 of Dr. Wei's exact 5 bullet points from Slide 9 for full credit.",
    },
    {
        "mod": "M15",
        "title": "Ursell Number Definition & Physical Interpretation (M15 Slides 10–11 & L15-P2)",
        "keywords": "ursell number ur quantify nonlinearity transition 20 26 stokes cnoidal l15-p2 consistent",
        "question": "How is the Ursell number defined, and how is it used to interpret wave nonlinearity and select wave theories?",
        "answer": "The Ursell number quantifies the degree of shallow-water nonlinearity and is defined as wave steepness divided by relative depth cubed: U_r = (H/L) / (h/L)³ = (H · L²) / h³. A small Ursell number (U_r ≪ 20–26) indicates weak shallow-water nonlinearity, corresponding to linear or Stokes-type waves (where nonlinearity is governed by wave steepness H/L in deep or intermediate water). A large Ursell number (U_r > 26) indicates strong shallow-water nonlinearity, corresponding to cnoidal or solitary wave theories. The range U_r ≈ 20–26 represents an approximate transition rather than a strict boundary.",
        "equations": "U_r = (H / L) / (h / L)³ = (H · L²) / h³   (Transition: U_r ≈ 20–26)",
        "trap": "In L15-P2(d), state explicitly: 'The Ursell number is small, indicating weak shallow-water nonlinearity. It is consistent with selecting a Stokes-type theory.'",
    },
    {
        "mod": "M15",
        "title": "Two Approaches to Developing Nonlinear (Finite-Amplitude) Wave Theories (M15 Slides 12–13)",
        "keywords": "two approaches finite amplitude nonlinear wave theories analytical perturbation numerical stream function stokes cnoidal",
        "question": "What are the two approaches used to develop finite-amplitude (nonlinear) wave theories, and which theories belong to each?",
        "answer": "Finite-amplitude wave theories relax the requirement that either H/h or H/L be small, and are formulated in two ways: (1) Analytical Approach: Analytically solving the velocity potential (φ) boundary value problem using successive approximations or perturbation power-series techniques — examples include Stokes Wave Theory (power series in wave steepness H/L, applicable in deep and intermediate water where H/h is small) and Cnoidal Wave Theory (Korteweg & de Vries 1895 KdV equations using Jacobi elliptic 'cn' functions, applicable in shallow water with finite H/h). (2) Numerical Approach: Solving the Laplace equation and nonlinear boundary conditions numerically — specifically Dean's (1965) Stream Function Wave Theory (using stream function ψ related to φ via the Cauchy-Riemann equations).",
        "equations": "Analytical: Stokes (H/L series) & Cnoidal (KdV 'cn');   Numerical: Stream Function ψ (Dean 1965)",
        "trap": "Also remember Slide 2: 'Linear wave theory is also called small-amplitude wave theory; Nonlinear wave theory is also called finite-amplitude wave theory.'",
    },
    {
        "mod": "M15",
        "title": "Stokes 2nd-Order Dispersion Relation & Surface Profile Asymmetry (M15 Slides 15–17)",
        "keywords": "stokes second order dispersion relation surface profile crest trough reinforce oppose vertical asymmetry flatter steeper",
        "question": "How does the Stokes 2nd-order surface profile differ from linear wave theory, and does its dispersion relation change?",
        "answer": "For Stokes second-order wave theory, the dispersion relation (σ² = gk tanh kh) is identical to linear wave theory (a nonlinear dispersion correction only appears at third order and above). However, the Stokes 2nd-order surface profile adds a second-harmonic correction term proportional to wave steepness (H/L) with twice the frequency cos 2(kx - σt). The first-order and second-order components reinforce each other (are in phase) at the wave crest and oppose each other at the wave trough. This produces vertical asymmetry: wave crests are steeper and more peaked (higher above SWL), while wave troughs are flatter and shallower than a linear cosine wave.",
        "equations": "η = (H/2)cos(kx - σt) + (πH/8)(H/L)[cosh(kh)/sinh³(kh)](2 + cosh 2kh)cos 2(kx - σt)",
        "trap": "At zero-crossings x = L/4 and 3L/4 (at t=0), cos(kx)=0 so linear η = 0, but cos(2kx) = cos(π) = -1, so Stokes 2nd-order η is NEGATIVE (-A₂)!",
    },
    {
        "mod": "M15",
        "title": "Cnoidal Wave Theory Limits & Stream Function Cauchy-Riemann Equations (M15 Slides 22–25)",
        "keywords": "cnoidal wave theory kdv korteweg de vries jacobi elliptic cn limits deep shallow solitary stream function cauchy riemann",
        "question": "What are the deep-water and shallow-water limits of Cnoidal wave theory, and how is Stream Function theory formulated?",
        "answer": "Cnoidal wave theory (Korteweg & de Vries, 1895, or KdV theory) expresses the wave profile using the Jacobi elliptic function 'cn'. Its deep-water limit is small-amplitude (linear) wave theory, and its shallow-water limit (as wavelength L → ∞) is solitary wave theory. Stream function wave theory (Dean, 1965) solves fully nonlinear water waves numerically using the stream function (ψ), which is related to velocity potential (φ) by the Cauchy-Riemann equations: u = -∂φ/∂x = -∂ψ/∂z and w = -∂φ/∂z = ∂ψ/∂x.",
        "equations": "Cnoidal limits: Linear (deep) ↔ Solitary (shallow);   Cauchy-Riemann: u = -∂φ/∂x = -∂ψ/∂z,  w = -∂φ/∂z = ∂ψ/∂x",
        "trap": "'Cnoidal' comes from the Jacobi elliptic function notation 'cn' used by Korteweg & de Vries (1895).",
    },
    {
        "mod": "M16",
        "title": "What is a Solitary Wave? (Oscillatory vs. Translatory) (M16 Slides 3–6)",
        "keywords": "what is solitary wave crest above swl no trough translatory oscillatory infinite wavelength period sech",
        "question": "What is a solitary wave, and why is it classified as a translatory wave rather than an oscillatory wave?",
        "answer": "A solitary wave is a nonlinear finite-amplitude shallow-water wave whose profile consists of a single crest lying completely above the still water level (SWL) with no wave trough (η ≥ 0 everywhere). Unlike linear periodic waves (which are oscillatory because water particles move in closed orbits), a solitary wave is a translatory wave: as the wave passes, water particles move forward in the direction of wave propagation and then come to rest without returning to complete an orbit. Theoretically, a solitary wave has an infinite wavelength (L → ∞) and an infinite wave period (T → ∞).",
        "equations": "η(x,t) = H · sech²[√(3H / 4h³) · (x - Ct)],   C = √(g(h + H)),   sech(x) = 1/cosh(x) = 2/(e^x + e^-x)",
        "trap": "Solitary wave speed C = √(g(h + H)) increases with wave height H, making it faster than linear shallow-water wave speed C = √(gh).",
    },
    {
        "mod": "M16",
        "title": "Solitary Wave Displaced Volume, Energy & SI Units (M16 Slides 9–10 & L16-P1)",
        "keywords": "solitary wave displaced volume total energy per unit crest width si units gamma h/h",
        "question": "How are the displaced water volume and total energy per unit crest width calculated for a solitary wave, and what are their SI units?",
        "answer": "For a solitary wave of height H in water depth h, the displaced volume of water per unit crest width above SWL is V = (16/3 · h³ · H)^(1/2), which has SI units of m² (m³ of volume per m of crest width). The total wave energy per unit crest width is evenly divided between kinetic energy (KE) and potential energy (PE) and is given by E = (8/3) · ρ · g · h³ · γ · √(γ / 3), where γ = H / h is the relative wave height. The SI unit of solitary wave total energy per unit crest width E is kg·m/s² (or J/m = N).",
        "equations": "V = √(16/3 · h³ · H) [m²],   E = (8/3)ρgh³γ√(γ/3) [kg·m/s² or J/m],   γ = H/h",
        "trap": "Unit Trap on Slide 9: Solitary wave energy E is per unit crest width (integrated along x), so its unit is kg·m/s² (J/m), whereas linear wave energy density E (M07) is per unit horizontal area [kg/s² or J/m²]!",
    },
    {
        "mod": "M16",
        "title": "What is a Tsunami & Why Never Say 'Tsunami Wave'? (M16 Slides 11–17)",
        "keywords": "what is tsunami japanese harbor wave never say tsunami wave causes earthquake 88% magnitude 8.0 landslide subaerial submarine ring of fire 78%",
        "question": "What does the word 'tsunami' mean, what causes tsunamis, how do earthquakes and landslides generate them, and where do they occur?",
        "answer": "(1) Meaning: 'Tsunami' comes from the Japanese characters for harbor ('tsu') and wave ('nami'), meaning 'harbor wave.' Do not use the term 'tsunami wave' because 'tsunami' already means wave. (2) Causes: Tsunamis are caused by large, sudden displacements of the ocean due to underwater earthquakes (most common), landslides, volcanic activity, weather, or near-earth objects; 88% of historical tsunamis in the NOAA database were generated by earthquakes or earthquake-triggered landslides. (3) Earthquake mechanism: At a tectonic subduction zone, a stuck overriding plate bulges under strain (tectonic uplift) and suddenly ruptures/slips, causing subsidence and releasing energy into the water column; generally, an earthquake must exceed Magnitude 8.0 to generate a dangerous distant tsunami. (4) Landslide mechanisms (2 ways): Subaerial (a landslide enters water from above) or Submarine (water is displaced ahead of and behind an underwater landslide). (5) Global distribution (1900–2015): 78% Pacific Ocean ('Ring of Fire'), 8% Atlantic & Caribbean, 6% Mediterranean, 5% Indian Ocean, 1% other seas.",
        "equations": "88% Earthquake/Landslide-driven | Earthquake M > 8.0 | Distribution: 78% Pacific, 8% Atlantic/Caribbean, 6% Med, 5% Indian",
        "trap": "Dr. Wei explicitly states on Slide 11: 'Don't use the term tsunami wave since tsunami has a meaning of wave.'",
    },
    {
        "mod": "M16",
        "title": "Why is a Tsunami a Shallow-Water Wave in the Deep Ocean & Green's Law (M16 Slide 18 & L16-P2)",
        "keywords": "tsunami shallow water wave deep ocean 6 km greens law shoaling h1/h2 1/4 assumptions speed wavelength height",
        "question": "Why does a tsunami behave as a shallow-water wave even in a 6 km deep ocean, and what happens as it shoals into coastal water (Green's Law)?",
        "answer": "A tsunami behaves as a shallow-water wave even in the deep ocean (e.g., h₁ = 6 km) because its wave period is extremely long (typically 10–60 minutes, such as T = 45 min = 2700 s), producing an enormous wavelength (L₁ = C₁·T ≈ 655 km). Consequently, the relative water depth is h₁/L₁ ≈ 6 / 655 = 0.009 < 0.05 (the shallow-water threshold). Assuming 1D propagation, slowly varying depth, no reflection, and no energy dissipation, conservation of energy flux ([EC_g]₁ = [EC_g]₂ with C_g = C = √(gh)) yields Green's Law: H₂ / H₁ = K_s = (h₁ / h₂)^(1/4). As the tsunami propagates from deep ocean into shallower coastal water, its wave speed C decreases, its wavelength L decreases, and its wave height H increases substantially.",
        "equations": "h₁/L₁ < 0.05 (Shallow-Water Wave),   C = √(gh) [or √(g(h+H))],   L = C·T,   H₂ = H₁ · (h₁ / h₂)^(1/4) [Green's Law]",
        "trap": "Include the 4 assumptions of Green's Law from Slide 18: (1) One-dimensional propagation, (2) Slowly varying depth, (3) No reflection, and (4) No energy dissipation.",
    },
    {
        "mod": "M14",
        "title": "Radiation Stress Gradients & Momentum Balance in Region 1 vs. Region 2 (M14 Slides 7–18)",
        "keywords": "region 1 region 2 radiation stress gradient dsxx/dx > 0 < 0 momentum balance wave set-down set-up storm surge",
        "question": "How does radiation stress S_xx change from deep water to the shoreline (Region 1 vs. Region 2), and how does horizontal momentum balance cause wave set-down and wave set-up?",
        "answer": "The horizontal momentum balance for a stationary 2D water column requires that the cross-shore gradient of radiation stress be balanced by the hydrostatic pressure gradient of the mean water level (MWL): dS_xx/dx + ρg(h + η̄)(dη̄/dx) = 0, or dη̄/dx = -(1 / ρgh)(dS_xx/dx). (1) Region 1 (Shoaling Zone — deep water to the breaking point): As waves shoal without breaking, wave height H, wave energy E, and group factor n all increase, so radiation stress increases toward shore (∂S_xx/∂x > 0). To balance this positive momentum flux gradient, the MWL slope must be negative (dη̄/dx < 0), causing wave set-down (a lowering of mean water level below SWL, η̄ < 0). (2) Region 2 (Surf Zone — breaking point to wave uprush limit): Once waves break, energy dissipates rapidly and H = γh decreases toward shore, so radiation stress decreases (∂S_xx/∂x < 0). To balance this negative gradient, the MWL slope must be positive (dη̄/dx > 0), causing wave set-up (a rise in mean water level above SWL toward the beach).",
        "equations": "dS_xx/dx + ρg(h + η̄)(dη̄/dx) = 0  ⇒  dη̄/dx = -(1/ρgh)(dS_xx/dx);   Region 1: ∂S_xx/∂x > 0 ⇒ set-down (η̄ < 0);   Region 2: ∂S_xx/∂x < 0 ⇒ set-up (dη̄/dx > 0)",
        "trap": "Two derivation steps (Slide 17): (1) Establish the force-momentum flux balance for a water column in a stationary situation, and (2) Apply the linear wave assumption and ignore higher-order terms.",
    },
    {
        "mod": "M13",
        "title": "Definition, Physical Meaning & Derivation of Radiation Stress (M13 Slides 8–19)",
        "keywords": "radiation stress definition excess flow of momentum longuet-higgins stewart 1964 mass energy momentum derivation",
        "question": "What is radiation stress, do linear waves transport mass, energy, and momentum, and what are the steps to derive radiation stress?",
        "answer": "According to linear wave theory, waves do NOT transport mass (because water particles move in closed orbital trajectories), but they DO transport wave energy (via work done by dynamic pressure) and wave momentum (the product of fluid mass density ρ and particle velocity u, yielding momentum flux ρu²). Radiation stress (Longuet-Higgins & Stewart, 1964) is 'the excess flow of momentum due to the presence of waves' beyond hydrostatic still-water pressure. It is derived from linear wave theory by: (1) Integrating the total pressure p and particle momentum flux ρu² over the total water depth (from z = -h to z = η) and averaging over a wave period, and (2) Subtracting the depth-integrated hydrostatic pressure p₀ = -ρgz below the still water level (from z = -h to z = 0).",
        "equations": "S_xx = Overline[ ∫_{-h}^{η} (p + ρu²) dz ] - ∫_{-h}^{0} p₀ dz,   SI Unit: N/m or kg/s²",
        "trap": "In Cartesian coordinates, the first subscript letter indicates the momentum component and the second letter indicates the axis on which it is projected.",
    },
    {
        "mod": "M12",
        "title": "Why Waves Break in Deep Water vs. Shallow Water & Breaker Types (M12 Slides 10–27)",
        "keywords": "why does wave break deep water shallow water steepness 1/7 crest particle velocity u = c0 shoaling ks infinity depth limited 0.78 spilling plunging surging",
        "question": "Explain why waves break in deep water versus shallow water, and how breaker types are classified using the Iribarren parameter.",
        "answer": "Wave breaking is caused by the instability of the wave crest/front, via different mechanisms in deep and shallow water: (1) Deep Water (Steepness-Limited): Phase speed C₀ = gT/(2π) is independent of H, while crest particle velocity u is proportional to H; when H₀/L₀ ≥ 1/7, crest velocity u equals C₀ and the wave breaks. (2) Shallow Water (Depth-Limited): As h → 0, C = C_g = √(gh) decreases and K_s increases toward infinity; breaking is controlled by local water depth when H_b/h_b ≈ 0.78 (0.4–1.2). Using the surf similarity (Iribarren) parameter ξ₀ = tan(α)/√(H₀/L₀), breakers are classified as Spilling (ξ₀ < 0.5, mild slopes, occurs in deep & shallow water), Plunging (0.5 < ξ₀ < 3.3, curling tongue, occurs in deep & shallow water), or Surging/Collapsing (ξ₀ > 3.3, steep beach, occurs only in shallow water).",
        "equations": "Deep: H₀/L₀ < 1/7;   Shallow: H_b/h_b ≈ 0.78 (or Goda 2010);   ξ₀ = tan(α)/√(H₀/L₀)",
        "trap": "Goda (2010) incorporates beach slope s and relative breaking depth h_b/L₀, whereas H_b/h_b = 0.78 is a simple constant approximation.",
    },
]

# ==========================================
# 4A. MODULE 15: NONLINEAR WAVE THEORIES
# ==========================================
if selected_module == "🔥 M15: Nonlinear Wave Theories":
    st.markdown('<div class="main-title">Module 15: Nonlinear Wave Theories (Stokes 2nd-Order & Le Méhauté)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Stokes 2nd-Order Surface Profile η(x,t), Crest/Trough Asymmetry, Ursell Number (U_r), & Le Méhauté (1976) Wave Theory Selection</div>', unsafe_allow_html=True)

    m15_mode = st.radio(
        "Select Problem Type:",
        (
            "Stokes 2nd-Order Wave Surface Profile & 5-Point Table (L15-P1, L15-Ex2)",
            "Ursell Number (U_r) & Le Méhauté (1976) Wave Theory Selection (L15-P2, L15-Ex1, L15-Ex3)",
        ),
        horizontal=True,
    )

    if m15_mode.startswith("Stokes 2nd-Order"):
        c1, c2, c3, c4 = st.columns(4)
        h_in = c1.number_input("Water Depth h [m]", value=28.0, step=0.5)
        T_in = c2.number_input("Wave Period T [s]", value=7.0, step=0.5)
        H_in = c3.number_input("Wave Height H [m]", value=3.5, step=0.5)
        t_raw = c4.text_input("Time t [s] (e.g. 0, T, T/2)", value="0")

        d = solve_dispersion_relation(h_in, T_in, g_in, solver_method)
        L = d["L"]
        L0 = d["L0"]
        k = d["k"]
        kh = d["kh"]
        sigma = d["sigma"]
        t_val = float(eval(t_raw, {"T": T_in, "L": L, "h": h_in, "pi": np.pi}))

        A1 = H_in / 2.0
        A2 = (np.pi * H_in / 8.0) * (H_in / L) * (np.cosh(kh) / (np.sinh(kh) ** 3)) * (2.0 + np.cosh(2.0 * kh))

        pts = [
            ("0", 0.0),
            ("L/4", L / 4.0),
            ("L/2", L / 2.0),
            ("3L/4", 3.0 * L / 4.0),
            ("L", L),
        ]
        rows = []
        for frac_lbl, x_val in pts:
            phase = k * x_val - sigma * t_val
            eta1 = A1 * np.cos(phase)
            eta2 = A2 * np.cos(2.0 * phase)
            eta_stokes = eta1 + eta2
            rows.append({
                "x (fraction)": frac_lbl,
                "x (m)": f"{x_val:.3f}",
                "Small-amplitude wave theory η (m)": f"{eta1:.3f}",
                "2nd-Order Correction η₂ (m)": f"{eta2:+.3f}",
                "Stokes 2nd order wave theory η (m)": f"{eta_stokes:.3f}",
            })

        traps = [
            ("M15 Trap #1 — Stokes 2nd-Order Uses Linear Dispersion", f"Stokes 2nd-order uses σ² = gk tanh(kh), giving L = {L:.3f} m and k = {k:.3f} rad/m ({k:.4f} rad/m). Dispersion corrections only occur at 3rd order and above!"),
            ("M15 Trap #2 — Crest Reinforcement vs. Trough Opposition", f"At crests (x = 0, L), η₁ and η₂ reinforce: η = {A1:.3f} + {A2:.3f} = {A1+A2:.3f} m. At the trough (x = L/2), they oppose: η = -{A1:.3f} + {A2:.3f} = {-A1+A2:.3f} m. At x = L/4 and 3L/4, Linear η = 0.000 m while Stokes 2nd η = -{A2:.3f} m!"),
        ]
        ovals = [
            ("Deep Wavelength", "L₀", f"{L0:.3f}", "m"),
            ("Local Wavelength", "L", f"{L:.3f}", "m"),
            ("Wavenumber", "k", f"{k:.3f}", f"rad/m ({k:.4f})"),
            ("Dimensionless kh", "kh", f"{kh:.4f}", "-"),
            ("Linear Amplitude", "A₁ = H/2", f"{A1:.3f}", "m"),
            ("Stokes 2nd Amplitude", "A₂", f"{A2:.3f}", "m"),
        ]
        finals = [
            ("x = 0 and x = L (Wave Crest)", "η_crest", f"{A1+A2:.3f}", "m", f"Linear η₁ = {A1:.3f} m | Stokes 2nd = {A1+A2:.3f} m"),
            ("x = L/4 & 3L/4 (Zero-Crossings)", "η_L/4", f"{-A2:.3f}", "m", f"Linear η₁ = 0.000 m | Stokes 2nd = {-A2:.3f} m"),
            ("x = L/2 (Wave Trough)", "η_trough", f"{-A1+A2:.3f}", "m", f"Linear η₁ = {-A1:.3f} m | Stokes 2nd = {-A1+A2:.3f} m"),
            ("Wavelength & Wavenumber", "L, k", f"{L:.3f} m, {k:.3f}", "rad/m", "Matches Dr. Wei L15-P1 Answer Key"),
        ]
        explanation = (
            f"At h = {h_in} m and T = {T_in} s, the wavelength is L = {L:.3f} m and the wave number is k = {k:.3f} rad/m. "
            f"The linear amplitude is H/2 = {A1:.3f} m and the 2nd-order Stokes harmonic amplitude is A₂ = {A2:.3f} m. "
            f"The two components reinforce each other at the wave crest (η = {A1+A2:.3f} m at x = 0 and L) and oppose each other "
            f"at the wave trough (η = {-A1+A2:.3f} m at x = L/2), producing vertical asymmetry with steeper, more peaked wave crests "
            f"and flatter, shallower wave troughs than a linear cosine profile."
        )
        render_goodnotes_terminal(
            f"h = {h_in} m   T = {T_in} s   H = {H_in} m   t = {t_raw} ({t_val:.1f} s)   g = {g_in} m/s²",
            "Wavelength L, wave number k, and water surface profile η(x,t) for Linear and Stokes 2nd-Order wave theory at x = 0, L/4, L/2, 3L/4, and L",
            [
                "σ² = gk tanh(kh)   ⇒   L = L₀ tanh(2πh/L),   k = 2π/L",
                "η_linear(x,t) = (H / 2) cos(kx - σt)",
                "η_Stokes2(x,t) = (H/2)cos(kx - σt) + (πH/8)(H/L)[cosh(kh)/sinh³(kh)](2 + cosh 2kh)cos 2(kx - σt)",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )
        st.markdown("#### 📊 Complete 5-Point Surface Profile Table (Copy Directly Into Your Exam Sheet)")
        st.table(rows)
    else:
        c1, c2, c3 = st.columns(3)
        h_in = c1.number_input("Water Depth h [m]", value=150.0, step=0.5)
        T_in = c2.number_input("Wave Period T [s]", value=16.0, step=0.5)
        H_in = c3.number_input("Wave Height H [m]", value=10.0, step=0.5)

        d = solve_dispersion_relation(h_in, T_in, g_in, solver_method)
        L = d["L"]
        L0 = d["L0"]
        steep = H_in / L
        rel_depth = h_in / L
        rel_height = H_in / h_in
        Ur = (H_in * (L ** 2)) / (h_in ** 3)
        dim_H = H_in / (g_in * (T_in ** 2))
        dim_h = h_in / (g_in * (T_in ** 2))
        theory_name, ur_comment = classify_le_mehaute(h_in, T_in, H_in, g_in, L, L0)

        traps = [
            ("M15 Trap #3 — Le Méhauté (1976) Dimensionless Axes & Notation", f"Vertical axis: H/(gT²) = {dim_H:.3f} ({dim_H:.4f}); Horizontal axis: h/(gT²) = {dim_h:.2f} ({dim_h:.4f}). Remember Le Méhauté uses λ for L and τ for T!"),
            ("M15 Trap #4 — Ursell Number Transition (U_r ≈ 20–26)", f"U_r = (H·L²)/h³ = {Ur:.2f}. Small U_r (≪ 26) indicates weak shallow-water nonlinearity (Stokes/Linear regime); large U_r (> 26) indicates strong shallow-water nonlinearity (Cnoidal/Solitary regime)."),
        ]
        ovals = [
            ("Deep Wavelength", "L₀", f"{L0:.3f}", "m"),
            ("Local Wavelength", "L", f"{L:.3f}", "m"),
            ("Wave Steepness", "H/L", f"{steep:.4f}", "-"),
            ("Relative Depth", "h/L", f"{rel_depth:.4f}", "-"),
            ("Relative Height", "H/h", f"{rel_height:.4f}", "-"),
            ("Gravity Scale", "gT²", f"{g_in*(T_in**2):.2f}", "m"),
        ]
        finals = [
            ("a) Ursell Number", "U_r", f"{Ur:.2f}", "-", f"Exact 3-dec: {Ur:.3f}  [Formula: H·L²/h³]"),
            ("b1) Dimensionless Wave Height", "H/(gT²)", f"{dim_H:.3f}", "-", f"Exact: {dim_H:.5f} (Vertical Axis)"),
            ("b2) Dimensionless Water Depth", "h/(gT²)", f"{dim_h:.2f}", "-", f"Exact: {dim_h:.4f} (Horizontal Axis)"),
            ("c) Selected Wave Theory", "Theory", theory_name, "", "From Le Méhauté (1976) diagram"),
        ]
        explanation = (
            f"a) The Ursell number is about U_r = {Ur:.2f} ({Ur:.3f}).  "
            f"b) Dimensionless wave height H/(gT²) = {dim_H:.3f} ({dim_H:.4f}); dimensionless water depth h/(gT²) = {dim_h:.2f} ({dim_h:.4f}).  "
            f"c) {theory_name} should be applied.  "
            f"d) {ur_comment}"
        )
        render_goodnotes_terminal(
            f"h = {h_in} m   T = {T_in} s   H = {H_in} m   g = {g_in} m/s²",
            "a) Ursell number U_r;  b) Dimensionless wave height H/(gT²) & depth h/(gT²);  c) Wave theory from Le Méhauté (1976);  d) Consistency explanation",
            [
                "L₀ = (gT²)/(2π),   L = L₀ tanh(2πh/L)",
                "U_r = (H / L) / (h / L)³ = (H · L²) / h³",
                "Dimensionless Wave Height = H / (gT²),   Dimensionless Water Depth = h / (gT²)",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )

# ==========================================
# 4B. MODULE 16: SOLITARY WAVES & TSUNAMIS
# ==========================================
elif selected_module == "🔥 M16: Solitary Waves & Tsunamis":
    st.markdown('<div class="main-title">Module 16: Solitary Wave Theory & Tsunamis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Solitary Wave Speed C = √(g(h+H)), Displaced Volume V, Total Energy E, & Tsunami Shoaling via Green’s Law</div>', unsafe_allow_html=True)

    m16_mode = st.radio(
        "Select Problem Type:",
        (
            "Solitary Wave Speed, Displaced Volume & Total Energy (L16-P1, L16-Ex1, L16-Ex2)",
            "Tsunami Propagation & Green's Law Shoaling (L16-P2, L16-Ex2, L16-Ex3)",
        ),
        horizontal=True,
    )

    if m16_mode.startswith("Solitary Wave"):
        c1, c2 = st.columns(2)
        H_in = c1.number_input("Solitary Wave Height H [m]", value=1.5, step=0.5)
        h_in = c2.number_input("Water Depth h [m]", value=15.0, step=0.5)

        C_sol = np.sqrt(g_in * (h_in + H_in))
        C_shal = np.sqrt(g_in * h_in)
        V_sol = np.sqrt((16.0 / 3.0) * (h_in ** 3) * H_in)
        gamma = H_in / h_in
        E_sol = (8.0 / 3.0) * rho_in * g_in * (h_in ** 3) * gamma * np.sqrt(gamma / 3.0)

        traps = [
            ("M16 Trap #1 — SI Units of V [m²] and E [kg·m/s² or J/m]", "Displaced volume per unit crest width V has SI units of m² (m³/m), and total energy per unit crest width E has SI units of kg·m/s² (or J/m = N)."),
            ("M16 Trap #2 — Solitary Speed C = √(g(h+H)) vs. Shallow Speed C = √(gh)", f"Solitary wave speed includes wave height H: C = √(g(h+H)) = {C_sol:.2f} m/s vs. linear shallow-water speed √(gh) = {C_shal:.2f} m/s."),
        ]
        ovals = [
            ("Total Crest Depth", "h + H", f"{h_in + H_in:.2f}", "m"),
            ("Relative Height", "γ = H/h", f"{gamma:.4f}", "-"),
            ("Energy Root Factor", "√(γ/3)", f"{np.sqrt(gamma/3.0):.4f}", "-"),
            ("Volume Radicand", "(16/3)h³H", f"{(16.0/3.0)*(h_in**3)*H_in:.1f}", "m⁴"),
        ]
        finals = [
            ("a1) Solitary-Wave Speed", "C", f"{C_sol:.2f}", "m/s", f"Exact 3-dec: {C_sol:.3f} m/s  [√(g(h+H))]"),
            ("a2) Shallow-Water Wave Speed", "C_shallow", f"{C_shal:.2f}", "m/s", f"Exact 3-dec: {C_shal:.3f} m/s  [√(gh)]"),
            ("b) Volume per Unit Crest Width", "V", f"{V_sol:.2f}", "m²", f"Exact 3-dec: {V_sol:.3f} m²  [√((16/3)h³H)]"),
            ("c) Total Energy per Crest Width", "E", f"{E_sol:.2f}", "kg·m/s²", f"Evenly split: KE = PE = {E_sol/2.0:.2f} kg·m/s² (J/m)"),
        ]
        explanation = (
            f"a) Solitary-wave speed: C = √[g(h + H)] = {C_sol:.2f} m/s, and shallow-water wave speed: C_shallow = √(gh) = {C_shal:.2f} m/s "
            f"(the solitary wave travels faster because its phase speed increases with wave amplitude H).  "
            f"b) Displaced volume per unit crest width: V = [(16/3)h³H]^(1/2) = {V_sol:.2f} m².  "
            f"c) With γ = H/h = {gamma:.3f}, total energy per unit crest width: E = (8/3)ρgh³γ√(γ/3) = {E_sol:.2f} kg·m/s²."
        )
        render_goodnotes_terminal(
            f"H = {H_in} m   h = {h_in} m   ρ = {rho_in} kg/m³   g = {g_in} m/s²",
            "a) Solitary-wave speed C & comparison with shallow-water speed;  b) Displaced volume per unit crest width V;  c) Total energy per unit crest width E",
            [
                "C = √[g(h + H)],   C_shallow = √(gh)",
                "V = [ (16 / 3) · h³ · H ]^(1/2)   [SI unit: m²]",
                "E = (8 / 3) · ρ · g · h³ · γ · √(γ / 3)   where γ = H / h   [SI unit: kg·m/s²]",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        T_min = c1.number_input("Tsunami Period T [minutes]", value=45.0, step=0.5, help="Converted automatically to seconds (45 min = 2700 s)")
        H1_in = c2.number_input("Offshore Tsunami Height H₁ [m]", value=0.75, step=0.25)
        h1_km = c3.number_input("Offshore Ocean Depth h₁ [km]", value=6.0, step=0.5, help="Converted automatically to meters (6 km = 6000 m)")
        h2_m = c4.number_input("Nearshore Water Depth h₂ [m]", value=20.0, step=0.5)

        T_sec = T_min * 60.0
        h1_m = h1_km * 1000.0
        Ks_green = (h1_m / h2_m) ** 0.25
        H2_m = H1_in * Ks_green

        # Dr. Wei 16p.pdf Answer Key convention (uses C = sqrt(g*(h+H)))
        C1_key = np.sqrt(g_in * (h1_m + H1_in))
        L1_key = C1_key * T_sec
        rel_h1 = h1_m / L1_key
        C2_key = np.sqrt(g_in * (h2_m + H2_m))
        L2_key = C2_key * T_sec

        # Pure Linear Shallow-Water formula (Slide 18: C = sqrt(g*h))
        C1_lin = np.sqrt(g_in * h1_m)
        L1_lin = C1_lin * T_sec
        C2_lin = np.sqrt(g_in * h2_m)
        L2_lin = C2_lin * T_sec

        traps = [
            ("M16 Trap #3 — CRITICAL L16-P2 Answer Key Discovery (√(g(h+H)) vs. √(gh))", f"In 16p.pdf, Dr. Wei's key computed H₂ = {H2_m:.2f} m via Green's Law and used C = √(g(h+H)) to get C₁ = {C1_key:.2f} m/s, L₁ = {L1_key:.1f} m, C₂ = {C2_key:.2f} m/s, and L₂ = {L2_key:.2f} m! Meanwhile, Slide 18 writes C = √(gh) (giving C₁ = {C1_lin:.2f} m/s, L₁ = {L1_lin:.1f} m, C₂ = {C2_lin:.2f} m/s, L₂ = {L2_lin:.2f} m). Both are shown below—write BOTH on your exam sheet for guaranteed 100% credit!"),
            ("M16 Trap #4 — Unit Conversions (min → s and km → m)", f"T = {T_min} min = {T_sec:.0f} s, and h₁ = {h1_km} km = {h1_m:.0f} m."),
        ]
        ovals = [
            ("Period in Seconds", "T", f"{T_sec:.0f}", "s"),
            ("Offshore Depth", "h₁", f"{h1_m:.0f}", "m"),
            ("Green's Law K_s", "(h₁/h₂)^(1/4)", f"{Ks_green:.4f}", "-"),
            ("Relative Depth", "h₁/L₁", f"{rel_h1:.3f}", "< 0.05 (Shallow)"),
            ("Linear √(gh₁)", "C₁,lin", f"{C1_lin:.2f}", f"m/s (L₁={L1_lin:.1f}m)"),
            ("Linear √(gh₂)", "C₂,lin", f"{C2_lin:.2f}", f"m/s (L₂={L2_lin:.2f}m)"),
        ]
        finals = [
            ("a1) Offshore Wave Speed C₁", "C₁", f"{C1_key:.2f}", "m/s", f"16p.pdf Key √(g(h₁+H₁)) = {C1_key:.2f} m/s | Linear √(gh₁) = {C1_lin:.2f} m/s"),
            ("a2) Offshore Wavelength L₁", "L₁", f"{L1_key:.1f}", "m", f"h₁/L₁ = {rel_h1:.3f} < 0.05 (Shallow Wave! Linear L₁ = {L1_lin:.1f} m)"),
            ("c) Nearshore Tsunami Height H₂", "H₂", f"{H2_m:.2f}", "m", f"Green's Law: H₁(h₁/h₂)^(1/4) = {H2_m:.3f} m"),
            ("b) Nearshore Speed C₂ & Wavelength L₂", "C₂, L₂", f"{C2_key:.2f} m/s, {L2_key:.2f}", "m", f"Linear √(gh₂): C₂ = {C2_lin:.2f} m/s, L₂ = {L2_lin:.2f} m"),
        ]
        explanation = (
            f"Offshore location (h₁ = {h1_m:.0f} m): Wave speed C₁ = {C1_key:.2f} m/s (or {C1_lin:.2f} m/s via √gh₁) and wavelength "
            f"L₁ = {L1_key:.1f} m (or {L1_lin:.1f} m). Relative depth h₁/L₁ = {rel_h1:.3f}, which is smaller than 0.05, verifying that the tsunami "
            f"behaves as a shallow-water wave even though the local ocean depth is {h1_km} km.  "
            f"Nearshore location (h₂ = {h2_m} m): By Green's Law, wave height H₂ = H₁(h₁/h₂)^(1/4) = {H2_m:.2f} m, wave speed C₂ = {C2_key:.2f} m/s "
            f"(or {C2_lin:.2f} m/s via √gh₂), and wavelength L₂ = {L2_key:.2f} m (or {L2_lin:.2f} m).  "
            f"The comparison shows that as the tsunami propagates into shallower water, wave speed decreases, wavelength decreases, and wave height increases."
        )
        render_goodnotes_terminal(
            f"T = {T_min} min ({T_sec:.0f} s)   H₁ = {H1_in} m   h₁ = {h1_km} km ({h1_m:.0f} m)   h₂ = {h2_m} m   g = {g_in} m/s²",
            "a) C₁, L₁ at 6 km & shallow-water verification;  b) C₂, L₂ at 20 m;  c) Nearshore wave height H₂;  d) Comparison of C, L, and H",
            [
                "C = √(gh)  [or Solitary Key: C = √(g(h + H))],   L = C · T,   Shallow-water check: h₁ / L₁ < 0.05",
                "[EC_g]₁ = [EC_g]₂   ⇒   H₂ / H₁ = K_s = (C_g1 / C_g2)^(1/2) = (h₁ / h₂)^(1/4)   [Green's Law]",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )

# ==========================================
# 4C. OPEN-ENDED EXAM Q&A SYNTHESIZER
# ==========================================
elif selected_module == "🧠 Open-Ended Exam Q&A Synthesizer":
    st.markdown('<div class="main-title">🧠 Dr. Wei Open-Ended Exam Question Synthesizer (Modules 07–16)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Type any conceptual or open-ended exam question to get the exact Dr. Wei lecture slide answer, required equations, and grading traps.</div>', unsafe_allow_html=True)

    col_q, col_f, col_k = st.columns([3, 1, 1.5])
    query_str = col_q.text_input("🔍 Type or Paste Your Open-Ended Exam Question:", placeholder="e.g. Why does wave break in deep water vs shallow water? Why don't we say tsunami wave? How to derive radiation stress?")
    mod_filter = col_f.selectbox("Filter by Module:", ("ALL", "M15", "M16", "M14", "M13", "M12"))
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key_in = col_k.text_input("✨ Gemini API Key (Optional):", value=default_key, type="password", help="Offline synthesis works automatically. Paste an AIza... Gemini API key to also synthesize custom answers with Gemini 2.5.")

    if st.button("✨ Synthesize Custom Answer with High-Level Gemini AI (Grounded in EGN 330 Slides)"):
        if not query_str.strip():
            st.warning("Please type an open-ended exam question above first.")
        elif not api_key_in.strip():
            st.info("No Gemini API key entered — using Instant Pre-Loaded EGN 330 Slide Synthesizer below (paste an AIza... key above for live cloud synthesis).")
        else:
            import json, urllib.request
            digest = "\n".join([f"[{x['mod']}] {x['title']}: Q: {x['question']} | A: {x['answer']} | Eq: {x['equations']}" for x in EGN330_QA_BANK])
            prompt = (
                "You are Thalassa, the EGN 330 Coastal Wave Mechanics Exam Assistant for Dr. Zhangping (John) Wei at UNCW. "
                "Answer the student's open-ended exam question concisely and using Dr. Wei's exact slide terminology, equations, and physical reasoning for 100% full credit. "
                "Provide:\n1) DIRECT EXAM PARAGRAPH TO WRITE DOWN (2-4 crisp sentences)\n2) KEY EQUATIONS & SYMBOLS TO INCLUDE\n3) EXAM GRADING TRAP TO AVOID\n\n"
                f"Course Lecture Knowledge Base:\n{digest}\n\nStudent Exam Question: {query_str}"
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key_in.strip()}"
            req = urllib.request.Request(
                url,
                data=json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.15}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    ai_text = resp_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response.")
                    st.markdown("#### ✨ Live Gemini 2.5 Synthesis (Grounded in EGN 330 Slides)")
                    st.markdown(f'<div class="gn-grey-box">{ai_text}</div>', unsafe_allow_html=True)
                    st.code(ai_text, language=None)
            except Exception as e:
                st.error(f"Gemini API Error: {e}. Using Instant Pre-Loaded Slide Synthesizer below.")

    tokens = [t for t in query_str.lower().split() if len(t) > 1]
    scored = []
    for item in EGN330_QA_BANK:
        if mod_filter != "ALL" and item["mod"] != mod_filter:
            continue
        if not tokens:
            scored.append((1, item))
        else:
            hay = f"{item['title']} {item['keywords']} {item['question']} {item['answer']} {item['equations']}".lower()
            s = 25 if query_str.lower() in hay else 0
            for tk in tokens:
                if tk in hay:
                    s += 5
            if s > 0:
                scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [x[1] for x in scored] if scored else EGN330_QA_BANK

    if results:
        top = results[0]
        st.markdown("#### 🎯 Top Full-Credit Exam Answer (Copy Directly Into Your Exam Sheet)")
        st.markdown(f'<div class="gn-grey-box"><strong>[{top["mod"]} — {top["title"]}]</strong><br><strong>Q:</strong> {top["question"]}<br><br><strong>FULL-CREDIT EXAM ANSWER:</strong><br>{top["answer"]}<br><br><strong>EQUATIONS TO WRITE:</strong> {top["equations"]}</div>', unsafe_allow_html=True)
        st.code(top['answer'] + chr(10) + chr(10) + "Equations: " + top['equations'], language=None)

    st.markdown("#### 📚 Matching EGN 330 Lecture Slide Topics")
    for it in results:
        with st.expander(f"[{it['mod']}] {it['title']} — {it['question']}", expanded=(len(results) <= 4)):
            st.markdown(f'<div class="gn-grey-box">{it["answer"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gn-formula-item"><strong>Required Equations:</strong> {it["equations"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="trap-alert"><strong>⚠️ Grading Trap:</strong> {it["trap"]}</div>', unsafe_allow_html=True)
            st.code(it["answer"], language=None)

# ==========================================
# 4D. MODULE 14: WAVE SET-UP & SET-DOWN
# ==========================================
elif selected_module == "🔥 M14: Wave Set-Up & Set-Down":
    st.markdown('<div class="main-title">Module 14: Wave Set-Up & Wave Set-Down</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Shoaling Wave Set-Down (η̄), Komar & Gaughan (1973) + Weggel (1972) Breaking Depth (h_b), and Surf-Zone Linear Wave Set-Up Function η̄(h)</div>', unsafe_allow_html=True)

    m14_mode = st.radio(
        "Select Problem Type:",
        (
            "Local Wave Set-Down at Depth h (L14-P1, L14-Ex1)",
            "Breaking (H_b, h_b, x_b) + Set-Down at Breaking + Linear Wave Set-Up Function (L14-P2, L14-Ex2)",
        ),
        horizontal=True,
    )

    if m14_mode.startswith("Local Wave Set-Down"):
        c1, c2, c3 = st.columns(3)
        H_in = c1.number_input("Wave Height H [m]", value=1.5, step=0.5)
        T_in = c2.number_input("Wave Period T [s]", value=6.0, step=0.5)
        h_in = c3.number_input("Water Depth h [m]", value=8.0, step=0.5)

        d = solve_dispersion_relation(h_in, T_in, g_in, solver_method)
        sinh_2kh = np.sinh(2.0 * d["kh"])
        eta_down = - (1.0 / 8.0) * (H_in ** 2) * d["k"] / sinh_2kh
        eta_down_cm = eta_down * 100.0
        eta_shallow = - (1.0 / 16.0) * (H_in ** 2) / h_in

        traps = [
            ("M14 Trap #1 — Negative Sign & Units (m vs. cm)", f"Wave set-down is ALWAYS negative (η̄ < 0) because ∂S_xx/∂x > 0 lowers the mean water level before breaking. Note both meters ({eta_down:.4f} m) and centimeters ({eta_down_cm:.2f} cm)!"),
            ("M14 Trap #2 — Independent of Bottom Slope", "Before breaking, wave set-down η̄ = -(1/8)(H²k)/sinh(2kh) depends strictly on local depth h, wave height H, and wavenumber k."),
        ]
        ovals = [
            ("Deep Wavelength", "L₀", f"{d['L0']:.3f}", "m"),
            ("Local Wavelength", "L", f"{d['L']:.3f}", "m"),
            ("Wavenumber", "k", f"{d['k']:.4f}", "rad/m"),
            ("Dimensionless 2kh", "2kh", f"{2*d['kh']:.4f}", "-"),
            ("Hyperbolic Denom", "sinh(2kh)", f"{sinh_2kh:.4f}", "-"),
        ]
        finals = [
            ("Wave Set-Down (Meters)", "η̄", f"{eta_down:.4f}", "m", f"3-dec: {eta_down:.3f} m | Shallow approx: {eta_shallow:.4f} m"),
            ("Wave Set-Down (Centimeters)", "η̄", f"{eta_down_cm:.2f}", "cm", f"Mean water level drops {abs(eta_down_cm):.2f} cm below SWL"),
        ]
        explanation = (
            f"At water depth h = {h_in} m (where L = {d['L']:.3f} m and k = {d['k']:.4f} rad/m), the wave set-down is "
            f"η̄ = {eta_down:.4f} m (or {eta_down_cm:.2f} cm). The minus sign indicates that wave shoaling increases radiation stress "
            f"S_xx toward shore (∂S_xx/∂x > 0), lowering the mean water level below the still water level (SWL)."
        )
        render_goodnotes_terminal(
            f"H = {H_in} m   T = {T_in} s   h = {h_in} m   g = {g_in} m/s²",
            "Local wavelength L, wavenumber k, and wave set-down η̄ at this location",
            [
                "L₀ = (gT²)/(2π),   L = L₀ tanh(2πh/L),   k = 2π/L",
                "dS_xx/dx + ρg(h + η̄)(dη̄/dx) = 0   ⇒   dη̄/dx = -(1 / ρgh)(dS_xx / dx)",
                "η̄ = - (1 / 8) · (H² · k) / sinh(2kh)",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        T_in = c1.number_input("Wave Period T [s]", value=15.0, step=0.5)
        H0_in = c2.number_input("Deep-Water Height H₀ [m]", value=1.5, step=0.5)
        slope_raw = c3.text_input("Beach Slope tan(β) (e.g. 1/20, 1/50, 0.05)", value="1/20")
        gamma_in = c4.number_input("Surf-Zone Breaker Index γ [-]", value=0.78, step=0.02)

        s_clean = slope_raw.strip().replace(":", "/")
        tan_beta = float(eval(s_clean))

        d0 = solve_dispersion_relation(None, T_in, g_in, solver_method)
        L0 = d0["L0"]
        omega_b = 0.56 * ((H0_in / L0) ** (-0.2))
        Hb = omega_b * H0_in
        Hb_2d = round(Hb, 2)

        a_weg = 43.8 * (1.0 - np.exp(-19.0 * tan_beta))
        b_weg = 1.56 / (1.0 + np.exp(-19.5 * tan_beta))
        gamma_b_weg = b_weg - a_weg * (Hb / (g_in * (T_in ** 2)))
        hb = Hb / gamma_b_weg
        hb_2d = round(hb, 2)
        xb = hb / tan_beta

        eta_b = - (1.0 / 16.0) * (Hb ** 2) / hb
        eta_b_key = - (1.0 / 16.0) * (Hb_2d ** 2) / hb_2d
        eta_b_2d = round(eta_b_key, 2)

        term_g = (3.0 / 8.0) * (gamma_in ** 2)
        K = term_g / (1.0 + term_g)
        K_2d = round(K, 2)
        eta_shore_max = eta_b_2d + K_2d * hb_2d

        traps = [
            ("M14 Trap #3 — Komar & Gaughan (1973) + Weggel (1972) for Breaking Point", f"Use Komar & Gaughan (1973) to find H_b = {Hb_2d:.2f} m and Weggel (1972) (γ_b = {gamma_b_weg:.3f}) to find h_b = {hb_2d:.2f} m. Do NOT use the surf-zone γ = {gamma_in} to find h_b!"),
            ("M14 Trap #4 — Surf-Zone Linear Set-Up Function", f"Use γ = {gamma_in} to compute K = (3/8 γ²)/(1 + 3/8 γ²) = {K_2d:.2f} ({K:.3f}), then write η̄(h) = η̄_b + K(h_b - h)."),
        ]
        ovals = [
            ("Deep Wavelength", "L₀", f"{L0:.3f}", "m"),
            ("Komar Index", "Ω_b", f"{omega_b:.3f}", "-"),
            ("Weggel Coeff a", "a", f"{a_weg:.3f}", "-"),
            ("Weggel Coeff b", "b", f"{b_weg:.3f}", "-"),
            ("Weggel γ_b", "γ_b", f"{gamma_b_weg:.3f}", "-"),
            ("Slope Factor K", "K", f"{K:.3f}", f"({K_2d:.2f})"),
        ]
        finals = [
            ("a1) Incipient Breaking Height", "H_b", f"{Hb_2d:.2f}", "m", f"Exact 3-dec: {Hb:.3f} m (Komar & Gaughan 1973)"),
            ("a2) Breaking Depth & Distance", "h_b", f"{hb_2d:.2f}", "m", f"Exact h_b = {hb:.3f} m (Weggel 1972) | x_b = {xb:.2f} m"),
            ("b) Wave Set-Down at Breaking", "η̄_b", f"{eta_b_2d:.2f}", "m", f"Exact 3-dec: {eta_b:.3f} m  [-(1/16)H_b²/h_b]"),
            ("c) Linear Wave Set-Up Function", "η̄(h)", f"{eta_b_2d:.2f} + {K_2d:.2f}*({hb_2d:.2f} - h)", "m", f"K = {K_2d:.2f} ({K:.3f}) | Shoreline (h=0) η̄_max = +{eta_shore_max:.3f} m"),
        ]
        explanation = (
            f"a) Using Komar & Gaughan (1973), the incipient breaking wave height is H_b = {Hb_2d:.2f} m ({Hb:.3f} m). "
            f"Using Weggel (1972) with a = {a_weg:.3f}, b = {b_weg:.3f}, and γ_b = {gamma_b_weg:.3f}, the breaking water depth is "
            f"h_b = {hb_2d:.2f} m ({hb:.3f} m) at offshore distance x_b = h_b/tan(β) = {xb:.2f} m.  "
            f"b) Wave set-down at the breaking location is η̄_b = -(1/16)(H_b²/h_b) = {eta_b_2d:.2f} m ({eta_b:.3f} m).  "
            f"c) With surf-zone breaker index γ = {gamma_in}, factor K = (3/8 γ²)/(1 + 3/8 γ²) = {K_2d:.2f} ({K:.3f}), giving the linear wave set-up function: "
            f"η̄ = {eta_b_2d:.2f} + {K_2d:.2f}*({hb_2d:.2f} - h)."
        )
        render_goodnotes_terminal(
            f"T = {T_in} s   H₀ = {H0_in} m   tan(β) = {slope_raw} ({tan_beta:.4f})   γ = {gamma_in}   g = {g_in} m/s²",
            "a) H_b, h_b, and x_b at breaking;  b) Wave set-down η̄_b at breaking;  c) Factor K and linear wave set-up function η̄(h)",
            [
                "H_b = 0.56 · H₀ · (H₀ / L₀)^(-1/5)   [Komar & Gaughan, 1973]",
                "γ_b = b - a·[H_b / (gT²)],   a = 43.8(1 - e^(-19 tan β)),   b = 1.56 / (1 + e^(-19.5 tan β)),   h_b = H_b / γ_b   [Weggel, 1972]",
                "η̄_b = - (1 / 16) · (H_b² / h_b),   K = (3/8 γ²) / (1 + 3/8 γ²),   η̄(h) = η̄_b + K · (h_b - h)",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )

# ==========================================
# 5. MODULE 12: WAVE BREAKING & RUN-UP
# ==========================================
elif selected_module == "🔥 M12: Wave Breaking & Run-Up":
    st.markdown('<div class="main-title">Module 12: Wave Breaking & Wave Run-Up</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Iribarren Parameter (ξ₀), Breaker Classification, Goda (2010), Komar & Gaughan (1973), and Stockdon (2006) 2% Run-Up</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    T_in = c1.number_input("Wave Period T [s]", value=15.0, step=0.5)
    H0_in = c2.number_input("Deep-Water Height H₀ [m]", value=1.5, step=0.5)
    hb_in = c3.number_input("Breaking Depth h_b [m]", value=3.0, step=0.5)
    slope_mode = c4.selectbox("Beach Slope Input Format", ("Degrees α [deg] (e.g. 15, 10, 1.5)", "Ratio s = tan(α) (e.g. 1/25, 1/10, 0.04)"))

    slope_raw = st.text_input("Beach Slope Value(s) — Enter single value (e.g. 1/25 or 0.1) or comma-separated list (e.g. 15, 10, 1.5):", value="15, 10, 1.5")

    disp0 = solve_dispersion_relation(None, T_in, g_in, solver_method)
    L0 = disp0["L0"]
    steep0 = H0_in / L0
    sqrt_steep0 = np.sqrt(steep0)

    parts = [p.strip() for p in slope_raw.split(",") if p.strip()]
    slope_items = []
    for p in parts:
        val = float(eval(p.replace(":", "/")))
        is_deg = slope_mode.startswith("Degrees")
        s_val = np.tan(np.radians(val)) if is_deg else val
        xi0 = s_val / sqrt_steep0
        if xi0 > 3.3: b_type = "Surging or collapsing (ξ₀ > 3.3)"
        elif xi0 >= 0.5: b_type = "Plunging (0.5 < ξ₀ < 3.3)"
        else: b_type = "Spilling (ξ₀ < 0.5)"
        slope_items.append((p, val, is_deg, s_val, xi0, b_type))

    s_primary = slope_items[0][3] if slope_items else 0.04
    xi_primary = slope_items[0][4] if slope_items else 0.26

    hb_L0 = hb_in / L0
    slope_factor = 1.0 + 11.0 * (s_primary ** (4.0 / 3.0))
    gamma_goda = (0.17 / hb_L0) * (1.0 - np.exp(-1.5 * np.pi * hb_L0 * slope_factor))
    Hb_goda = gamma_goda * hb_in
    Hb_simple = 0.78 * hb_in
    gamma_kamp = 0.56 * np.exp(3.5 * s_primary)
    Hb_kamp = gamma_kamp * hb_in
    omega_komar = 0.56 * (steep0 ** (-0.2))
    Hb_komar = omega_komar * H0_in
    H0_max_deep = L0 / 7.0

    sqrt_H0L0 = np.sqrt(H0_in * L0)
    ru_low = 0.043 * sqrt_H0L0
    ru_high = 1.1 * (0.35 * s_primary * sqrt_H0L0 + np.sqrt(H0_in * L0 * (0.563 * (s_primary ** 2) + 0.004)) / 2.0)
    ru_2pct = ru_low if xi_primary < 0.3 else ru_high

    traps = [
        ("Trap #3 — Slope Format Guard (deg vs. ratio)", f"Active Format: {slope_mode}. Remember: L12-P1 gives degrees (15°, 10°, 1.5° → take tan α), while L12-P2 (1/25 = 0.04) and L12-P3 (1/10 = 0.10) already give s = tan(α)!"),
        ("Stockdon Cutoff (ξ₀ = 0.3) vs. Breaker Cutoff (ξ₀ = 0.5)", "Breaker type changes from Spilling to Plunging at ξ₀ = 0.5, whereas Stockdon 2% wave run-up switches equations at ξ₀ = 0.3!"),
    ]

    ovals = [
        ("Deep Wavelength", "L₀", f"{L0:.3f}", "m"),
        ("Deep Steepness", "H₀/L₀", f"{steep0:.5f}", "-"),
        ("Steepness Root", "√(H₀/L₀)", f"{sqrt_steep0:.4f}", "-"),
        ("Relative Depth", "h_b/L₀", f"{hb_L0:.4f}", "-"),
        ("Goda Slope Term", "1+11s^(4/3)", f"{slope_factor:.4f}", "-"),
        ("Run-up Scale", "√(H₀L₀)", f"{sqrt_H0L0:.3f}", "m"),
    ]
    for p, val, is_deg, s_val, xi0, b_type in slope_items:
        ovals.append((f"Slope ({p}{'°' if is_deg else ''})", "tan(α)", f"{s_val:.4f}", "-"))

    finals = []
    for p, val, is_deg, s_val, xi0, b_type in slope_items:
        finals.append((f"Iribarren & Breaker (Slope {p}{'°' if is_deg else ''})", "ξ₀", f"{xi0:.3f}", f"→ {b_type}", f"tan(α) = {s_val:.4f}"))

    finals.extend([
        ("Stockdon (2006) 2% Wave Run-Up", "R_u2%", f"{ru_2pct:.3f}", "m", f"Regime: {'ξ₀ < 0.3 (Dissipative)' if xi_primary < 0.3 else 'ξ₀ > 0.3 (Full Eq)'} | Low={ru_low:.3f}m, High={ru_high:.3f}m"),
        ("Goda (2010) Breaker Index", "H_b / h_b", f"{gamma_goda:.3f}", "-", f"Compared to simple constant 0.78 (Kamphuis = {gamma_kamp:.3f})"),
        ("Goda (2010) Breaking Wave Height", "H_b", f"{Hb_goda:.3f}", "m", f"At h_b = {hb_in} m (Simple 0.78·h_b = {Hb_simple:.3f} m)"),
        ("Komar & Gaughan (1973) Incipient Height", "H_b,inc", f"{Hb_komar:.3f}", "m", f"Ω_b = {omega_komar:.3f} | Deep max H₀,max = L₀/7 = {H0_max_deep:.3f} m"),
    ])

    explanation = (
        f"For T = {T_in} s (L₀ = {L0:.3f} m), "
        + " ".join([f"slope {p}{'°' if is_deg else ''} gives ξ₀ = {xi0:.3f} ({b_type});" for p, val, is_deg, s_val, xi0, b_type in slope_items])
        + f" Stockdon (2006) uses the {'dissipative (ξ₀ < 0.3)' if xi_primary < 0.3 else 'reflective/intermediate (ξ₀ > 0.3)'} equation to give R_u2% = {ru_2pct:.3f} m. "
        + f"At h_b = {hb_in} m, Goda (2010) gives H_b/h_b = {gamma_goda:.3f} and H_b = {Hb_goda:.3f} m, which differs from the constant 0.78 ({Hb_simple:.3f} m) because Goda accounts for beach slope (s = {s_primary:.3f}) and relative depth (h_b/L₀ = {hb_L0:.4f})."
    )

    render_goodnotes_terminal(
        f"T = {T_in} s   H₀ = {H0_in} m   h_b = {hb_in} m   Slope = {slope_raw}   g = {g_in} m/s²",
        "L₀, ξ₀ & Breaker Type, Stockdon 2% Run-Up (R_u2%), and Goda (2010) Breaker Index (γ_b) & Breaking Height (H_b)",
        [
            "L₀ = (g · T²) / (2π) ≈ 1.5613 · T²",
            "ξ₀ = tan(α) / √(H₀ / L₀)",
            "Stockdon (ξ₀ < 0.3): R_u2% = 0.043 √(H₀L₀)  |  (ξ₀ > 0.3): R_u2% = 1.1[0.35 tan(α)√(H₀L₀) + √(H₀L₀(0.563 tan²α + 0.004))/2]",
            "Goda (2010): H_b / h_b = [0.17 / (h_b/L₀)] · {1 - exp[-1.5π (h_b/L₀)(1 + 11 s^(4/3))]}",
        ],
        ovals,
        finals,
        explanation,
        traps,
    )

# ==========================================
# 6. MODULE 13: WAVE RADIATION STRESS
# ==========================================
elif selected_module == "🔥 M13: Wave Radiation Stress":
    st.markdown('<div class="main-title">Module 13: Wave Radiation Stress</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Normal & Oblique Excess Momentum Flux Components (S_xx, S_yy, S_xy, S_yx) + Two-Depth Shoaling Coupling</div>', unsafe_allow_html=True)

    m13_mode = st.radio(
        "Select Problem Type:",
        ("Single Depth h (L13-P1, L13-P2, Ex2)", "Shoaling h₁ → h₂ + Radiation Stress (L13-P3, Ex3)", "Pure Deep Water n = 0.5 (L13-Ex1)"),
        horizontal=True,
    )

    if "Shoaling" in m13_mode:
        c1, c2, c3, c4, c5 = st.columns(5)
        H1 = c1.number_input("Offshore Height H₁ [m]", value=0.5, step=0.5)
        T = c2.number_input("Wave Period T [s]", value=8.0, step=0.5)
        h1 = c3.number_input("Offshore Depth h₁ [m] (0 = Deep)", value=7.0, step=0.5)
        h2 = c4.number_input("Nearshore Depth h₂ [m]", value=3.0, step=0.5)
        theta_deg = c5.number_input("Angle to x-axis θ [deg]", value=0.0, step=0.5)

        d1 = solve_dispersion_relation(None if h1 <= 0 else h1, T, g_in, solver_method)
        d2 = solve_dispersion_relation(h2, T, g_in, solver_method)
        Ks = np.sqrt(d1["Cg"] / d2["Cg"])
        H2 = H1 * Ks

        th_rad = np.radians(theta_deg)
        E1 = (1.0 / 8.0) * rho_in * g_in * (H1 ** 2)
        Sxx1 = E1 * (d1["n"] * (np.cos(th_rad) ** 2 + 1.0) - 0.5)
        Syy1 = E1 * (d1["n"] * (np.sin(th_rad) ** 2 + 1.0) - 0.5)
        Sxy1 = E1 * d1["n"] * np.sin(th_rad) * np.cos(th_rad)

        E2 = (1.0 / 8.0) * rho_in * g_in * (H2 ** 2)
        Sxx2 = E2 * (d2["n"] * (np.cos(th_rad) ** 2 + 1.0) - 0.5)
        Syy2 = E2 * (d2["n"] * (np.sin(th_rad) ** 2 + 1.0) - 0.5)
        Sxy2 = E2 * d2["n"] * np.sin(th_rad) * np.cos(th_rad)

        traps = [
            ("Trap #1 — Radiation Stress Units", "Radiation stress is depth-integrated (N/m² × m = N/m = kg/s²). Always write kg/s² (or N/m), never N/m²!"),
            ("Trap #4 — Hidden Shoaling Step Executed", f"Wave height increases from H₁ = {H1:.3f} m to H₂ = {H2:.3f} m (K_s = {Ks:.3f}) before computing nearshore E₂ and S_ij!"),
        ]
        ovals = [
            ("Deep Wavelength", "L₀", f"{d1['L0']:.3f}", "m"),
            ("Offshore Wavelength", "L₁", f"{d1['L']:.3f}", "m"),
            ("Offshore Group Factor", "n₁", f"{d1['n']:.3f}", "-"),
            ("Offshore Group Speed", "Cg₁", f"{d1['Cg']:.3f}", "m/s"),
            ("Offshore Energy", "E₁", f"{E1:.3f}", "kg/s²"),
            ("Shoaling Coeff", "K_s", f"{Ks:.3f}", "-"),
            ("Shoaled Height", "H₂", f"{H2:.3f}", "m"),
            ("Nearshore Wavelength", "L₂", f"{d2['L']:.3f}", "m"),
            ("Nearshore Wavenumber", "k₂", f"{d2['k']:.3f}", "rad/m"),
            ("Nearshore Group Factor", "n₂", f"{d2['n']:.3f}", "-"),
            ("Nearshore Group Speed", "Cg₂", f"{d2['Cg']:.3f}", "m/s"),
            ("Nearshore Energy", "E₂", f"{E2:.3f}", "kg/s²"),
        ]
        finals = [
            (f"Offshore (h₁={h1}m) Normal x-x", "S_xx,1", f"{Sxx1:.3f}", "kg/s²", f"E₁ = {E1:.3f} kg/s², n₁ = {d1['n']:.3f}"),
            (f"Offshore (h₁={h1}m) Normal y-y", "S_yy,1", f"{Syy1:.3f}", "kg/s²", f"Shear S_xy,1 = S_yx,1 = {Sxy1:.3f} kg/s²"),
            (f"Nearshore (h₂={h2}m) Normal x-x", "S_xx,2", f"{Sxx2:.3f}", "kg/s²", f"Using shoaled H₂ = {H2:.3f} m, E₂ = {E2:.3f} kg/s²"),
            (f"Nearshore (h₂={h2}m) Normal y-y", "S_yy,2", f"{Syy2:.3f}", "kg/s²", f"Shear S_xy,2 = S_yx,2 = {Sxy2:.3f} kg/s²"),
        ]
        explanation = (
            f"Both normal components (S_xx and S_yy) increase as the wave shoals from h₁ = {h1} m to h₂ = {h2} m "
            f"(S_xx: {Sxx1:.3f} → {Sxx2:.3f} kg/s²; S_yy: {Syy1:.3f} → {Syy2:.3f} kg/s²) because shoaling increases wave height "
            f"(H: {H1:.3f} → {H2:.3f} m) and therefore energy density (E: {E1:.3f} → {E2:.3f} kg/s²), while shallower depth also increases "
            f"the group velocity factor (n: {d1['n']:.3f} → {d2['n']:.3f}). Shear components remain {Sxy2:.3f} kg/s²."
        )
        render_goodnotes_terminal(
            f"H₁ = {H1} m   T = {T} s   h₁ = {h1} m   h₂ = {h2} m   θ = {theta_deg}°   ρ = {rho_in} kg/m³   g = {g_in} m/s²",
            "Offshore and Nearshore Radiation Stress Components (S_xx, S_yy, S_xy, S_yx) and Shoaling Comparison",
            [
                "K_s = √(Cg₁ / Cg₂),   H₂ = H₁ · K_s",
                "E = ⅛ · ρ · g · H²,   n = ½ [1 + (2kh) / sinh(2kh)]",
                "S_xx = E(2n - ½),   S_yy = E(n - ½),   S_xy = S_yx = 0  (for θ = 0°)",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )
    else:
        is_deep = "Pure Deep" in m13_mode
        c1, c2, c3, c4 = st.columns(4)
        H = c1.number_input("Wave Height H [m]", value=0.45, step=0.5)
        T = c2.number_input("Wave Period T [s]", value=9.0, step=0.5)
        h = c3.number_input("Water Depth h [m]", value=3.0, step=0.5, disabled=is_deep)
        theta_deg = c4.number_input("Wave Angle to x-axis θ [deg]", value=0.0, step=0.5)

        d = solve_dispersion_relation(None if is_deep else h, T, g_in, solver_method)
        E = (1.0 / 8.0) * rho_in * g_in * (H ** 2)
        n = 0.5 if is_deep else d["n"]
        th_rad = np.radians(theta_deg)
        cos2 = np.cos(th_rad) ** 2
        sin2 = np.sin(th_rad) ** 2
        Sxx = E * (n * (cos2 + 1.0) - 0.5)
        Syy = E * (n * (sin2 + 1.0) - 0.5)
        Sxy = E * n * np.sin(th_rad) * np.cos(th_rad)

        traps = [
            ("Trap #1 — Units of Radiation Stress", "Always write kg/s² (or N/m), identical to Wave Energy Density E — never N/m²!"),
            ("Trap #2 — Angle θ Convention", f"θ = {theta_deg}° is measured from the x-axis. {'Since θ = 75° > 45°, propagation is closer to the y-axis, so S_yy > S_xx!' if theta_deg > 45 else ''}"),
        ]
        ovals = [
            ("Deep Wavelength", "L₀", f"{d['L0']:.3f}", "m"),
            ("Local Wavelength", "L", f"{d['L']:.3f}", "m"),
            ("Wavenumber", "k", f"{d['k']:.3f}", "rad/m"),
            ("Wave Energy Density", "E", f"{E:.3f}", "kg/s²"),
            ("Group Factor", "n", f"{n:.3f}", "-"),
        ]
        finals = [
            ("Radiation Stress (x-x)", "S_xx", f"{Sxx:.3f}", "kg/s²", "Normal x-momentum flux"),
            ("Radiation Stress (y-y)", "S_yy", f"{Syy:.3f}", "kg/s²", "Normal y-momentum flux"),
            ("Radiation Stress (x-y)", "S_xy", f"{Sxy:.4f}", "kg/s²", f"3-dec: {Sxy:.3f} kg/s²"),
            ("Radiation Stress (y-x)", "S_yx", f"{Sxy:.4f}", "kg/s²", "Symmetric: S_yx = S_xy"),
        ]
        if abs(theta_deg) < 1e-6:
            explanation = f"Because the waves propagate along the x-axis (θ = 0°), S_xx ({Sxx:.3f} kg/s²) > S_yy ({Syy:.3f} kg/s²), and the shear components vanish (S_xy = S_yx = 0 kg/s²) because sin(0°) = 0."
        elif theta_deg > 45.0:
            explanation = f"Because θ = {theta_deg}° is much closer to the y-axis than to the x-axis, the larger normal momentum-flux component is in the y-direction (S_yy = {Syy:.3f} kg/s² > S_xx = {Sxx:.3f} kg/s²), while oblique propagation produces nonzero cross terms (S_xy = S_yx = {Sxy:.4f} kg/s²)."
        else:
            explanation = f"At oblique angle θ = {theta_deg}°, S_xx = {Sxx:.3f} kg/s², S_yy = {Syy:.3f} kg/s², and oblique momentum transfer produces nonzero shear stresses S_xy = S_yx = {Sxy:.4f} kg/s²."

        render_goodnotes_terminal(
            f"H = {H} m   T = {T} s   {'n = 0.5 (Deep)' if is_deep else f'h = {h} m'}   θ = {theta_deg}°   ρ = {rho_in} kg/m³   g = {g_in} m/s²",
            "Wavelength L, wavenumber k, wave energy E, group velocity factor n, and radiation stresses S_xx, S_yy, S_xy, S_yx",
            [
                "L₀ = (gT²)/(2π),   L = L₀ tanh(2πh/L),   k = 2π/L",
                "E = ⅛ ρ g H²,   n = ½ [1 + (2kh)/sinh(2kh)]",
                "S_xx = E[n(cos²θ + 1) - ½],   S_yy = E[n(sin²θ + 1) - ½],   S_xy = S_yx = E n sin(θ) cos(θ)",
            ],
            ovals,
            finals,
            explanation,
            traps,
        )

# ==========================================
# 7. MODULES M07 – M11 (WITH STEP = 0.5 & GOODNOTES UI)
# ==========================================
elif selected_module == "M07: Wave Energy & Power":
    st.markdown('<div class="main-title">Module 07: Wave Energy & Power Flux</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    h_in = c1.number_input("Depth h [m]", value=1.0, step=0.5)
    T_in = c2.number_input("Period T [s]", value=2.0, step=0.5)
    H_in = c3.number_input("Height H [m]", value=0.2, step=0.5)

    d = solve_dispersion_relation(h_in, T_in, g_in, solver_method)
    E = (1.0 / 8.0) * rho_in * g_in * (H_in ** 2)
    Ep, Ek = E / 2.0, E / 2.0
    F_bar = E * d["Cg"]

    render_goodnotes_terminal(
        f"h = {h_in} m   T = {T_in} s   H = {H_in} m   ρ = {rho_in} kg/m³   g = {g_in} m/s²",
        "Potential energy P̅E̅, kinetic energy K̅E̅, total wave energy E, group velocity Cg, and energy flux F̄",
        ["E = ⅛ ρ g H²   (P̅E̅ = K̅E̅ = E / 2)", "n = ½ [1 + (2kh)/sinh(2kh)],   Cg = n · C", "F̄ = E · Cg"],
        [
            ("Deep Wavelength", "L₀", f"{d['L0']:.3f}", "m"),
            ("Local Wavelength", "L", f"{d['L']:.3f}", "m"),
            ("Wavenumber", "k", f"{d['k']:.3f}", "rad/m"),
            ("Phase Speed", "C", f"{d['C']:.3f}", "m/s"),
            ("Group Factor", "n", f"{d['n']:.3f}", "-"),
        ],
        [
            ("Potential Energy", "P̅E̅", f"{Ep:.3f}", "kg/s²", "Equipartition: E / 2"),
            ("Kinetic Energy", "K̅E̅", f"{Ek:.3f}", "kg/s²", "Equipartition: E / 2"),
            ("Total Energy Density", "E", f"{E:.3f}", "kg/s² (J/m²)", "Independent of T and h!"),
            ("Group Velocity", "Cg", f"{d['Cg']:.3f}", "m/s", f"n · C = {d['n']:.3f} × {d['C']:.3f}"),
            ("Wave Energy Flux", "F̄", f"{F_bar:.3f}", "W/m (kg·m/s³)", f"{F_bar/1000:.3f} kW/m"),
        ],
        f"Total wave energy density E = {E:.3f} kg/s² depends strictly on wave height H (independent of period T and depth h). By equipartition, P̅E̅ = K̅E̅ = {Ep:.3f} kg/s².",
    )

elif selected_module == "M08: 1D Wave Shoaling":
    st.markdown('<div class="main-title">Module 08: 1D Wave Shoaling</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    H1 = c1.number_input("Offshore Height H₁ (or H₀) [m]", value=0.5, step=0.5)
    T = c2.number_input("Period T [s]", value=8.0, step=0.5)
    h1 = c3.number_input("Depth 1 h₁ [m] (0 = Deep Water)", value=7.0, step=0.5)
    h2 = c4.number_input("Depth 2 h₂ [m]", value=3.0, step=0.5)

    d1 = solve_dispersion_relation(None if h1 <= 0 else h1, T, g_in, solver_method)
    d2 = solve_dispersion_relation(h2, T, g_in, solver_method)
    Ks = np.sqrt(d1["Cg"] / d2["Cg"])
    H2 = H1 * Ks

    render_goodnotes_terminal(
        f"H₁ = {H1} m   T = {T} s   h₁ = {h1} m   h₂ = {h2} m   g = {g_in} m/s²",
        "Shoaling coefficient K_s and shoaled wave height H₂",
        ["L₀ = (gT²)/(2π),   L = L₀ tanh(2πh/L)", "K_s = √(Cg₁ / Cg₂),   H₂ = H₁ · K_s"],
        [
            ("Deep Wavelength", "L₀", f"{d1['L0']:.3f}", "m"),
            ("Offshore L₁", "L₁", f"{d1['L']:.3f}", "m"),
            ("Offshore Cg₁", "Cg₁", f"{d1['Cg']:.3f}", "m/s"),
            ("Nearshore L₂", "L₂", f"{d2['L']:.3f}", "m"),
            ("Nearshore n₂", "n₂", f"{d2['n']:.3f}", "-"),
            ("Nearshore Cg₂", "Cg₂", f"{d2['Cg']:.3f}", "m/s"),
        ],
        [
            ("Shoaling Coefficient", "K_s", f"{Ks:.3f}", "-", f"√({d1['Cg']:.3f} / {d2['Cg']:.3f})"),
            ("Shoaled Wave Height", "H₂", f"{H2:.3f}", "m", f"{H1} × {Ks:.3f}"),
        ],
        f"As waves shoal from h₁ = {h1} m to h₂ = {h2} m, group velocity decreases ({d1['Cg']:.3f} → {d2['Cg']:.3f} m/s), increasing wave height by K_s = {Ks:.3f} to H₂ = {H2:.3f} m.",
    )

elif selected_module == "M09: Shoaling & Refraction":
    st.markdown('<div class="main-title">Module 09: 2D Combined Shoaling & Refraction</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    H0 = c1.number_input("Deepwater H₀ [m]", value=1.0, step=0.5)
    T = c2.number_input("Period T [s]", value=15.0, step=0.5)
    alpha0 = c3.number_input("Deep Angle α₀ [deg]", value=75.0, step=0.5)
    h = c4.number_input("Local Depth h [m]", value=3.0, step=0.5)

    d = solve_dispersion_relation(h, T, g_in, solver_method)
    sin_a = (d["C"] / d["C0"]) * np.sin(np.radians(alpha0))
    alpha_rad = np.arcsin(np.clip(sin_a, -1.0, 1.0))
    alpha_deg = np.degrees(alpha_rad)
    Ks = np.sqrt(d["Cg0"] / d["Cg"])
    Kr = np.sqrt(np.cos(np.radians(alpha0)) / np.cos(alpha_rad))
    H = H0 * Ks * Kr

    render_goodnotes_terminal(
        f"T = {T} s   H₀ = {H0} m   α₀ = {alpha0}°   h = {h} m   g = {g_in} m/s²",
        "α, K_s, K_r, H, and effect dominance analysis",
        ["sin(α) = (C / C₀) sin(α₀)", "K_s = √(Cg₀ / Cg),   K_r = √(cos α₀ / cos α),   H = H₀ · K_s · K_r"],
        [
            ("Deep Wavelength", "L₀", f"{d['L0']:.2f}", "m"),
            ("Deep Celerity", "C₀", f"{d['C0']:.2f}", "m/s"),
            ("Deep Group Speed", "Cg₀", f"{d['Cg0']:.2f}", "m/s"),
            ("Local Wavelength", "L", f"{d['L']:.2f}", "m"),
            ("Local Celerity", "C", f"{d['C']:.2f}", "m/s"),
            ("Local Group Factor", "n", f"{d['n']:.3f}", "-"),
            ("Local Group Speed", "Cg", f"{d['Cg']:.2f}", "m/s"),
            ("Snell Ratio", "sin(α)", f"{sin_a:.4f}", "-"),
        ],
        [
            ("Refracted Angle", "α", f"{alpha_deg:.2f}", "°", f"arcsin({sin_a:.4f})"),
            ("Shoaling Coefficient", "K_s", f"{Ks:.3f}", "-", "Amplification factor"),
            ("Refraction Coefficient", "K_r", f"{Kr:.3f}", "-", "Crest spreading factor"),
            ("Resulting Wave Height", "H", f"{H:.3f}", "m", f"{H0} × {Ks:.3f} × {Kr:.3f}"),
        ],
        f"Despite K_s = {Ks:.3f} > 1, the resulting wave height H = {H:.3f} m is {'smaller' if H < H0 else 'larger'} than H₀ = {H0} m because K_r = {Kr:.3f}. Thus, {'refraction' if abs(H - H0) > 0 and H < H0 else 'shoaling'} has the larger influence on the final wave height.",
    )

elif selected_module == "M10: Standing Waves & Pressure":
    st.markdown('<div class="main-title">Module 10: Standing Wave Kinematics & Pressure</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    H = c1.number_input("Wave Height H [m]", value=1.5, step=0.5)
    T = c2.number_input("Period T [s]", value=5.0, step=0.5)
    h = c3.number_input("Depth h [m]", value=15.0, step=0.5)

    d = solve_dispersion_relation(h, T, g_in, solver_method)
    L, k, sigma = d["L"], d["k"], d["sigma"]
    c4, c5, c6 = st.columns(3)
    x_in = c4.text_input("Horizontal x [m] (e.g. L/3)", value="L/3")
    z_in = c5.text_input("Vertical z [m] (e.g. -h/3)", value="-h/3")
    t_in = c6.text_input("Time t [s] (e.g. 3*T/5)", value="3*T/5")

    x = float(eval(x_in.replace("L", str(L))))
    z = float(eval(z_in.replace("h", str(h))))
    t = float(eval(t_in.replace("T", str(T))))

    u = (H / 2) * (g_in * k / sigma) * (np.cosh(k * (h + z)) / np.cosh(k * h)) * np.sin(k * x) * np.sin(sigma * t)
    w = -(H / 2) * (g_in * k / sigma) * (np.sinh(k * (h + z)) / np.cosh(k * h)) * np.cos(k * x) * np.sin(sigma * t)
    ax = (H / 2) * (sigma ** 2) * (np.cosh(k * (h + z)) / np.sinh(k * h)) * np.sin(k * x) * np.cos(sigma * t)
    az = -(H / 2) * (sigma ** 2) * (np.sinh(k * (h + z)) / np.sinh(k * h)) * np.cos(k * x) * np.cos(sigma * t)
    theta = np.degrees(np.arctan(-np.tanh(k * (h + z)) / np.tan(k * x))) if np.tan(k * x) != 0 else 90.0
    Ph = -rho_in * g_in * z
    Pd = rho_in * g_in * (H / 2) * (np.cosh(k * (h + z)) / np.cosh(k * h)) * np.cos(k * x) * np.cos(sigma * t)
    P_tot = Ph + Pd

    render_goodnotes_terminal(
        f"T = {T} s   H = {H} m   h = {h} m   x₁ = {x_in} ({x:.2f} m)   z₁ = {z_in} ({z:.2f} m)   t = {t_in} ({t:.2f} s)",
        "Velocities (u, w), local accelerations (a_x, a_z), trajectory angle θ, and pressures (p_h, p_d, p_total)",
        [
            "u = (H/2)(gk/σ)[cosh k(h+z)/cosh kh] sin(kx) sin(σt),   w = -(H/2)(gk/σ)[sinh k(h+z)/cosh kh] cos(kx) sin(σt)",
            "a_x = (H/2)σ²[cosh k(h+z)/sinh kh] sin(kx) cos(σt),   a_z = -(H/2)σ²[sinh k(h+z)/sinh kh] cos(kx) cos(σt)",
            "tan(θ) = -tanh k(h+z) / tan(kx),   p_total = p_h + p_d",
        ],
        [
            ("Local Wavelength", "L", f"{L:.2f}", "m"),
            ("Wavenumber", "k", f"{k:.4f}", "rad/m"),
            ("Angular Freq", "σ", f"{sigma:.3f}", "rad/s"),
        ],
        [
            ("Horizontal Velocity", "u", f"{u:.3f}", "m/s", "-x dir" if u < 0 else "+x dir"),
            ("Vertical Velocity", "w", f"{w:.3f}", "m/s", "downward" if w < 0 else "upward"),
            ("Horizontal Accel", "a_x", f"{ax:.3f}", "m/s²", "-x dir" if ax < 0 else "+x dir"),
            ("Vertical Accel", "a_z", f"{az:.3f}", "m/s²", "downward" if az < 0 else "upward"),
            ("Trajectory Angle", "θ", f"{theta:.2f}", "°", "Particle path slope"),
            ("Total Pressure", "p_total", f"{P_tot:.1f}", "N/m²", f"p_h = {Ph:.1f} N/m² | p_d = {Pd:.1f} N/m²"),
        ],
        f"Since u {'< 0' if u < 0 else '> 0'} and w {'< 0' if w < 0 else '> 0'}, the instantaneous velocity vector points toward the {'-x' if u < 0 else '+x'} direction and {'downward' if w < 0 else 'upward'}. Since a_x {'< 0' if ax < 0 else '> 0'} and a_z {'< 0' if az < 0 else '> 0'}, the local acceleration vector points toward the {'-x' if ax < 0 else '+x'} direction and {'downward' if az < 0 else 'upward'}.",
    )

elif selected_module == "M11: Breakwater Diffraction":
    st.markdown('<div class="main-title">Module 11: Breakwater Diffraction</div>', unsafe_allow_html=True)
    scenario = st.radio("Scenario:", ("Semi-Infinite Breakwater (L11-P1)", "Breakwater Gap (L11-P2)"), horizontal=True)

    if scenario == "Semi-Infinite Breakwater (L11-P1)":
        c1, c2, c3 = st.columns(3)
        Hi = c1.number_input("Incident Height H_i [m]", value=3.0, step=0.5)
        T = c2.number_input("Period T [s]", value=8.0, step=0.5)
        h = c3.number_input("Depth h [m]", value=15.0, step=0.5)
        c4, c5, c6 = st.columns(3)
        r = c4.number_input("Distance r [m]", value=40.9, step=0.5)
        theta = c5.number_input("Incident Angle θ [deg]", value=75.0, step=0.5)
        beta = c6.number_input("Shadow Angle β [deg]", value=60.0, step=0.5)

        d = solve_dispersion_relation(h, T, g_in, solver_method)
        r_L = r / d["L"]
        Kd, sn_t, sn_rl, sn_b = lookup_wiegel_kd(theta, r_L, beta)
        Hd = Hi * Kd

        render_goodnotes_terminal(
            f"T = {T} s   h = {h} m   θ = {theta}°   β = {beta}°   r = {r} m   H_i = {Hi} m",
            "Diffracted wave height H_d at specified coordinates",
            ["L₀ = 1.56 T²,   L = L₀ tanh(2πh/L)", "r/L,   K_d = f(θ, β, r/L) [Wiegel 1962],   H_d = K_d · H_i"],
            [
                ("Deep Wavelength", "L₀", f"{d['L0']:.2f}", "m"),
                ("Local Wavelength", "L", f"{d['L']:.2f}", "m"),
                ("Relative Distance", "r/L", f"{r_L:.3f}", "-"),
                ("Wiegel Table K_d", "K_d", f"{Kd:.2f}", "-"),
            ],
            [
                ("Diffraction Coefficient", "K_d", f"{Kd:.2f}", "-", f"Wiegel (θ={sn_t}°, r/L={sn_rl}, β={sn_b}°)"),
                ("Diffracted Wave Height", "H_d", f"{Hd:.2f}", "m", f"{Kd:.2f} × {Hi}"),
            ],
            f"At r/L = {r_L:.3f} (≈ {sn_rl}), θ = {theta}°, and β = {beta}°, the Wiegel (1962) table yields K_d = {Kd:.2f}, giving a diffracted wave height of H_d = {Hd:.2f} m.",
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        Hi = c1.number_input("Incident H_i [m]", value=1.0, step=0.5)
        T = c2.number_input("Period T [s]", value=7.0, step=0.5)
        h = c3.number_input("Depth h [m]", value=10.0, step=0.5)
        B = c4.number_input("Gap Width B [m]", value=50.0, step=0.5)

        d = solve_dispersion_relation(h, T, g_in, solver_method)
        ratio = B / d["L"]
        Kd = 0.50 if ratio < 1.0 else (1.00 if ratio > 5.0 else 0.5 + 0.1 * ratio)
        Hd = Hi * Kd
        state_str = "B/L < 1 (Strong Diffraction)" if ratio < 1.0 else ("B/L > 5 (Minimal)" if ratio > 5.0 else "1 ≤ B/L ≤ 5 (Moderate)")

        render_goodnotes_terminal(
            f"H_i = {Hi} m   T = {T} s   B = {B} m   h = {h} m   g = {g_in} m/s²",
            "L, B/L, diffraction state assessment, K_d, and H_d",
            ["L₀ = 1.56 T²,   L = L₀ tanh(2πh/L)", "Gap Parameter = B / L,   H_d = K_d · H_i"],
            [
                ("Deep Wavelength", "L₀", f"{d['L0']:.2f}", "m"),
                ("Local Wavelength", "L", f"{d['L']:.2f}", "m"),
                ("Gap Ratio", "B/L", f"{ratio:.3f}", "-"),
            ],
            [
                ("Diffraction State", "State", state_str, "", f"B/L = {ratio:.3f}"),
                ("Diffraction Coeff", "K_d", f"{Kd:.2f}", "-", "Centerline approximation"),
                ("Diffracted Height", "H_d", f"{Hd:.2f}", "m", f"{Kd:.2f} × {Hi}"),
            ],
            f"Since B/L = {ratio:.3f} < 1, diffraction through the gap is strong. Using the centerline approximation K_d = {Kd:.2f}, the diffracted wave height is H_d = {Hd:.2f} m.",
        )
