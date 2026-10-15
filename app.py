"""
EGN 330: Coastal Wave Mechanics — Master Computational Engine (v3.0)
Fully Merged Implementation: Modules M07 through M13 (Exam-Compliant High-Contrast GoodNotes UI)
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
    page_title="EGN 330: Master Engine (M07–M13)",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #050507; color: #f8fafc; }
    .main-title { font-size: 1.85rem; font-weight: 800; color: #facc15; margin-bottom: 0.15rem; letter-spacing: -0.02em; }
    .sub-title { font-size: 0.95rem; color: #cbd5e1; margin-bottom: 1.0rem; font-weight: 500; }
    
    /* GoodNotes Top Knowns & Blue Find Box */
    .gn-knowns-box {
        background-color: #09090b;
        border: 2px solid #52525b;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: 'Courier New', monospace;
        font-size: 1.08rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .gn-find-box {
        background-color: #0c1e3d;
        border: 2.5px solid #38bdf8;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: 'Courier New', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: #e0f2fe;
        margin-bottom: 14px;
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
    
    /* GoodNotes Bottom Grey Explanation Box */
    .gn-grey-box {
        background-color: #1e293b;
        border: 2.5px solid #94a3b8;
        border-radius: 8px;
        padding: 14px 16px;
        color: #f8fafc;
        font-size: 1.02rem;
        line-height: 1.5;
        font-weight: 500;
        margin-top: 8px;
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

def render_goodnotes_terminal(knowns_str: str, find_str: str, formulas: list, ovals: list, finals: list, explanation: str, traps: list = None):
    if traps:
        for title, body in traps:
            st.markdown(f'<div class="trap-alert"><strong>⚠️ {title}:</strong> {body}</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([5, 7])
    with col_left:
        st.markdown("#### 1. GoodNotes Top Line & Blue 'Find' Box")
        st.markdown(f'<div class="gn-knowns-box">{knowns_str}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gn-find-box"><span style="color:#38bdf8;">Find:</span> {find_str}</div>', unsafe_allow_html=True)
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

        st.markdown("#### 5. GoodNotes Bottom Grey Explanation Box")
        st.markdown(f'<div class="gn-grey-box">{explanation}</div>', unsafe_allow_html=True)
        st.code(explanation, language=None)

# ==========================================
# 3. SIDEBAR NAVIGATION & GLOBAL CONSTANTS
# ==========================================
with st.sidebar:
    st.title("🌊 EGN 330 Engine v3.0")
    selected_module = st.radio(
        "Select Exam Module:",
        (
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
# 4. MODULE 12: WAVE BREAKING & RUN-UP
# ==========================================
if selected_module == "🔥 M12: Wave Breaking & Run-Up":
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
        val = float(eval(p))
        is_deg = slope_mode.startswith("Degrees")
        s_val = np.tan(np.radians(val)) if is_deg else val
        xi0 = s_val / sqrt_steep0
        if xi0 > 3.3: b_type = "Surging or collapsing (ξ₀ > 3.3)"
        elif xi0 >= 0.5: b_type = "Plunging (0.5 < ξ₀ < 3.3)"
        else: b_type = "Spilling (ξ₀ < 0.5)"
        slope_items.append((p, val, is_deg, s_val, xi0, b_type))

    s_primary = slope_items[0][3] if slope_items else 0.04
    xi_primary = slope_items[0][4] if slope_items else 0.26

    # Goda (2010), Kamphuis (1991), Simple (0.78), Komar & Gaughan (1973), Stockdon (2006)
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
# 5. MODULE 13: WAVE RADIATION STRESS
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
            f"H₁ = {H1} m   T = {T} s   h₁ = {h1} m → h₂ = {h2} m   θ = {theta_deg}°   ρ = {rho_in} kg/m³   g = {g_in} m/s²",
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
            f"H = {H} m   T = {T} s   {'Deep Water (n=0.5)' if is_deep else f'h = {h} m'}   θ = {theta_deg}°   ρ = {rho_in} kg/m³   g = {g_in} m/s²",
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
# 6. MODULES M07 – M11 (WITH STEP = 0.5 & GOODNOTES UI)
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
