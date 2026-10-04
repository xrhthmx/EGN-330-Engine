"""
EGN 330: Coastal Engineering Master Computational Engine
Fully Merged Implementation: Modules M07 through M11
"""
import streamlit as st
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import fresnel
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(page_title="EGN 330: Master Engine", page_icon="🌊", layout="wide")
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #94a3b8; margin-bottom: 1.5rem; }
    .abet-header { background: linear-gradient(90deg, #1e293b, #0f172a); padding: 12px 18px; border-left: 4px solid #38bdf8; border-radius: 4px; font-weight: 600; color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE MATHEMATICAL ENGINE (CACHED)
# ==========================================
@st.cache_data
def solve_dispersion_relation(h: float, T: float, method: str = "Exact Implicit", g: float = 9.81):
    sigma = 2.0 * np.pi / T
    L0 = g * (T ** 2) / (2.0 * np.pi)
    C0 = L0 / T
    if method == "Beji (2013) Explicit":
        u = (2.0 * np.pi * h) / L0
        denom = (u / np.sqrt(np.tanh(u))) * (1.0 + (u ** 1.09) * np.exp(-(1.55 + 1.3 * u + 0.216 * (u ** 2))))
        L = (2.0 * np.pi * h) / denom
        k = 2.0 * np.pi / L
    else:
        k0 = 2.0 * np.pi / L0
        y = k0 * h
        x = y * (1.0 + (y ** 1.09) * np.exp(-(1.55 + 1.3 * y + 0.216 * (y ** 2)))) ** (-0.5)
        k_seed = x / h
        def disp_func(k_val): return sigma ** 2 - g * k_val * np.tanh(k_val * h)
        def disp_fprime(k_val):
            th = np.tanh(k_val * h)
            return -g * (th + k_val * h * (1.0 - th ** 2))
        try:
            sol = root_scalar(disp_func, fprime=disp_fprime, x0=k_seed, method='newton', rtol=1e-12)
            k = sol.root
        except Exception:
            sol = root_scalar(disp_func, bracket=[0.0001, 10.0], method='brentq')
            k = sol.root
        L = 2.0 * np.pi / k
    C = L / T
    kh = k * h
    if kh > 15.0: n = 0.5
    elif kh < 0.05: n = 1.0
    else: n = 0.5 * (1.0 + (2.0 * kh) / np.sinh(2.0 * kh))
    Cg = n * C
    return {"L0": L0, "C0": C0, "L": L, "C": C, "k": k, "kh": kh, "n": n, "Cg": Cg}

def sommerfeld_Kd(r: float, theta_rad: float, alpha_rad: float, k: float):
    if r <= 0: return 1.0
    u1 = -2.0 * np.sqrt(k * r / np.pi) * np.sin(0.5 * (theta_rad - alpha_rad))
    u2 = -2.0 * np.sqrt(k * r / np.pi) * np.sin(0.5 * (theta_rad + alpha_rad))
    S1, C1 = fresnel(u1)
    S2, C2 = fresnel(u2)
    f1 = 0.5 * ((0.5 + C1) + 1j * (0.5 + S1))
    f2 = 0.5 * ((0.5 + C2) + 1j * (0.5 + S2))
    t1 = f1 * np.exp(-1j * k * r * np.cos(theta_rad - alpha_rad))
    t2 = f2 * np.exp(-1j * k * r * np.cos(theta_rad + alpha_rad))
    return float(np.abs(t1 + t2))

# ==========================================
# 3. ABET SO1 REUSABLE OUTPUT TERMINAL
# ==========================================
def render_abet_output(knowns: dict, unknowns: list, equations: list, calculations: list, final_answers: dict, module_title: str):
    st.markdown('<div class="abet-header">📋 ABET Student Outcome 1 Compliance Terminal</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Problem Identification")
        for k, v in knowns.items(): st.markdown(f"- `{k}`: **{v}**")
        st.subheader("2. Formulated Principles")
        for eq in equations: st.markdown(f"- {eq}")
    with col2:
        st.subheader("4. Accuracy of Solutions")
        for step in calculations: st.markdown(f"• {step}")
        st.markdown("### Final Answers:")
        for param, val in final_answers.items(): st.success(f"**{param}** = `{val}`")

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("EGN 330 Engine")
    selected_module = st.radio("Select Module:", ("M07: Wave Energy & Power", "M08: Wave Shoaling", "M09: Shoaling & Refraction", "M10: Standing Waves", "M11: Breakwater Diffraction"))
    dispersion_method = st.selectbox("Solver Method:", ("Exact Implicit", "Beji (2013) Explicit"))

# ==========================================
# 5. MODULE ROUTING
# ==========================================
if selected_module == "M07: Wave Energy & Power":
    st.markdown('<div class="main-title">Module 07: Wave Energy & Power</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    h_in = c1.number_input("Depth h [m]", value=10.0)
    T_in = c2.number_input("Period T [s]", value=8.0)
    H_in = c3.number_input("Height H [m]", value=1.5)
    
    disp = solve_dispersion_relation(h_in, T_in, method=dispersion_method)
    E = (1.0 / 8.0) * 1000.0 * 9.81 * (H_in ** 2)
    F_bar = E * disp["Cg"]
    
    m1, m2 = st.columns(2)
    m1.metric("Energy Density (E)", f"{E:,.2f} J/m²")
    m2.metric("Energy Flux (F̄)", f"{F_bar:,.2f} W/m")
    
    render_abet_output(
        {"h": f"{h_in} m", "T": f"{T_in} s", "H": f"{H_in} m"},
        ["E", "F_bar"],
        ["E = 1/8 * rho * g * H^2", "F_bar = E * Cg"],
        [f"Cg = {disp['Cg']:.3f} m/s", f"E = {E:.2f} J/m²", f"F_bar = {F_bar:.2f} W/m"],
        {"E": f"{E:.2f} J/m²", "F_bar": f"{F_bar:.2f} W/m"},
        "M07: Wave Energy"
    )

elif selected_module == "M08: Wave Shoaling":
    st.markdown('<div class="main-title">Module 08: Wave Shoaling</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    H0 = c1.number_input("Deepwater Height H0 [m]", value=2.0)
    T = c2.number_input("Period T [s]", value=10.0)
    h1 = c3.number_input("Depth 1 [m]", value=20.0)
    h2 = c4.number_input("Depth 2 [m]", value=4.0)
    
    disp1 = solve_dispersion_relation(h1, T, method=dispersion_method)
    disp2 = solve_dispersion_relation(h2, T, method=dispersion_method)
    Cg0 = 0.5 * disp1["C0"]
    Ks1, Ks2 = np.sqrt(Cg0 / disp1["Cg"]), np.sqrt(Cg0 / disp2["Cg"])
    H2 = H0 * Ks2
    
    m1, m2 = st.columns(2)
    m1.metric("Shoaling Coeff Ks2", f"{Ks2:.4f}")
    m2.metric("Shoaled Height H2", f"{H2:.3f} m")

    render_abet_output(
        {"H0": f"{H0} m", "T": f"{T} s", "h2": f"{h2} m"},
        ["Ks2", "H2"],
        ["Ks = sqrt(Cg0 / Cg)", "H2 = H0 * Ks2"],
        [f"Cg0 = {Cg0:.3f} m/s", f"Cg2 = {disp2['Cg']:.3f} m/s", f"Ks2 = {Ks2:.4f}"],
        {"Ks2": f"{Ks2:.4f}", "H2": f"{H2:.3f} m"},
        "M08: Wave Shoaling"
    )

elif selected_module == "M09: Shoaling & Refraction":
    st.markdown('<div class="main-title">Module 09: Shoaling & Refraction</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    H0 = c1.number_input("H0 [m]", value=2.5)
    T = c2.number_input("T [s]", value=8.0)
    alpha0 = c3.number_input("Deep Angle α0 [deg]", value=40.0)
    h = c4.number_input("Local Depth h [m]", value=6.0)
    
    disp = solve_dispersion_relation(h, T, method=dispersion_method)
    alpha_rad = np.arcsin((disp["C"] / disp["C0"]) * np.sin(np.radians(alpha0)))
    alpha_deg = np.degrees(alpha_rad)
    Ks = np.sqrt((0.5 * disp["C0"]) / disp["Cg"])
    Kr = np.sqrt(np.cos(np.radians(alpha0)) / np.cos(alpha_rad))
    H = H0 * Ks * Kr
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Refracted Angle α", f"{alpha_deg:.2f}°")
    m2.metric("Refraction Kr", f"{Kr:.4f}")
    m3.metric("Combined Height H", f"{H:.3f} m")

    render_abet_output(
        {"H0": f"{H0} m", "alpha0": f"{alpha0}°", "h": f"{h} m"},
        ["alpha", "Kr", "Ks", "H"],
        ["sin(alpha) = (C/C0)*sin(alpha0)", "Kr = sqrt(cos(alpha0)/cos(alpha))"],
        [f"alpha = {alpha_deg:.2f}°", f"Ks = {Ks:.4f}", f"Kr = {Kr:.4f}"],
        {"H": f"{H:.3f} m"},
        "M09: Shoaling & Refraction"
    )

elif selected_module == "M10: Standing Waves":
    st.markdown('<div class="main-title">Module 10: Standing Waves</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    Hi = c1.number_input("Incident Hi [m]", value=1.2)
    T = c2.number_input("T [s]", value=6.0)
    h = c3.number_input("Depth h [m]", value=8.0)
    
    disp = solve_dispersion_relation(h, T, method=dispersion_method)
    H_max = Hi * 2.0  # Assuming full reflection Kr = 1.0
    u_max = (2.0 * np.pi / T) * (H_max / (2 * np.sinh(disp["k"] * h)))
    
    m1, m2 = st.columns(2)
    m1.metric("Antinode Height Hmax", f"{H_max:.2f} m")
    m2.metric("Max Node Velocity u", f"{u_max:.3f} m/s")

    render_abet_output(
        {"Hi": f"{Hi} m", "h": f"{h} m"},
        ["Hmax", "u_max (node)"],
        ["Hmax = Hi(1+Kr)", "u = (Hmax/2)(gk/sigma)*..."],
        [f"k = {disp['k']:.4f}", f"Hmax = {H_max:.2f} m"],
        {"Hmax": f"{H_max:.2f} m", "u_max": f"{u_max:.3f} m/s"},
        "M10: Standing Waves"
    )

elif selected_module == "M11: Breakwater Diffraction":
    st.markdown('<div class="main-title">Module 11: Breakwater Diffraction</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    Hi = c1.number_input("Incident Hi [m]", value=3.0)
    T = c2.number_input("T [s]", value=8.0)
    h = c3.number_input("Depth h [m]", value=15.0)
    
    c4, c5, c6 = st.columns(3)
    r = c4.number_input("Distance r [m]", value=40.92)
    theta = c5.number_input("Angle θ [deg]", value=75.0)
    alpha = c6.number_input("Incident Angle α [deg]", value=60.0)
    
    disp = solve_dispersion_relation(h, T, method=dispersion_method)
    Kd = sommerfeld_Kd(r, np.radians(theta), np.radians(alpha), disp["k"])
    Hd = Hi * Kd
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Distance Ratio r/L", f"{(r/disp['L']):.3f}")
    m2.metric("Diffraction Coeff Kd", f"{Kd:.4f}")
    m3.metric("Diffracted Height Hd", f"{Hd:.3f} m")

    render_abet_output(
        {"Hi": f"{Hi} m", "r": f"{r} m", "theta": f"{theta}°"},
        ["Kd", "Hd"],
        ["Sommerfeld Analytical Fresnel Function", "Hd = Hi * Kd"],
        [f"k = {disp['k']:.4f}", f"r/L = {(r/disp['L']):.4f}"],
        {"Kd": f"{Kd:.4f}", "Hd": f"{Hd:.3f} m"},
        "M11: Breakwater Diffraction"
    )
