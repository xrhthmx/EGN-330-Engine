"""
EGN 330: Coastal Engineering Master Computational Engine
Fully Merged Implementation: Modules M07 through M11 (Exam Compliant)
"""
import streamlit as st
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import fresnel
import matplotlib.pyplot as plt

st.set_page_config(page_title="EGN 330: Master Engine", page_icon="🌊", layout="wide")
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.2rem; }
    .abet-header { background: linear-gradient(90deg, #1e293b, #0f172a); padding: 12px 18px; border-left: 4px solid #38bdf8; border-radius: 4px; font-weight: 600; color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def solve_dispersion_relation(h: float, T: float, g: float = 9.81):
    sigma = 2.0 * np.pi / T
    L0 = g * (T ** 2) / (2.0 * np.pi)
    C0 = L0 / T
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
    C = L / T
    kh = k * h
    if kh > 15.0: n = 0.5
    elif kh < 0.05: n = 1.0
    else: n = 0.5 * (1.0 + (2.0 * kh) / np.sinh(2.0 * kh))
    Cg = n * C
    return {"L0": L0, "C0": C0, "L": L, "C": C, "k": k, "kh": kh, "n": n, "Cg": Cg, "sigma": sigma}

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

def render_abet_output(knowns: dict, calculations: list, final_answers: dict):
    st.markdown('<div class="abet-header">📋 ABET Student Outcome 1 Compliance Terminal</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Problem Identification (Knowns)")
        for k, v in knowns.items(): st.markdown(f"- `{k}`: **{v}**")
        st.subheader("2. Intermediate Calculations")
        for step in calculations: st.markdown(f"• {step}")
    with col2:
        st.subheader("4. Final Accuracy Verification")
        for param, val in final_answers.items(): st.success(f"**{param}** = `{val}`")

with st.sidebar:
    st.title("🌊 EGN 330 Engine")
    selected_module = st.radio("Select Module:", ("M07: Wave Energy", "M08: Wave Shoaling", "M09: Refraction", "M10: Standing Waves", "M11: Diffraction"))
    rho_in = st.number_input("Water Density ρ [kg/m³]", value=1000.0, step=25.0, help="1000 for freshwater, 1025 for seawater")
    g_in = 9.81

if selected_module == "M07: Wave Energy":
    st.markdown('<div class="main-title">Module 07: Wave Energy & Power Flux</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    h_in = c1.number_input("Depth h [m]", value=1.0)
    T_in = c2.number_input("Period T [s]", value=2.0)
    H_in = c3.number_input("Height H [m]", value=0.2)
    
    disp = solve_dispersion_relation(h_in, T_in, g_in)
    E = (1.0 / 8.0) * rho_in * g_in * (H_in ** 2)
    Ep = E / 2.0
    Ek = E / 2.0
    F_bar = E * disp["Cg"]
    
    render_abet_output(
        {"h": f"{h_in} m", "T": f"{T_in} s", "H": f"{H_in} m", "ρ": f"{rho_in} kg/m³"},
        [f"Wavelength L = {disp['L']:.3f} m", f"Wave Speed C = {disp['C']:.2f} m/s", f"Group Factor n = {disp['n']:.2f}"],
        {"Potential Energy Ep": f"{Ep:.1f} kg/s²", "Kinetic Energy Ek": f"{Ek:.1f} kg/s²", "Total Energy E": f"{E:.2f} kg/s²", "Group Velocity Cg": f"{disp['Cg']:.2f} m/s", "Power Flux F_bar": f"{F_bar:.2f} W/m"}
    )

elif selected_module == "M08: Wave Shoaling":
    st.markdown('<div class="main-title">Module 08: Wave Shoaling</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    H1 = c1.number_input("Offshore Height H1 [m]", value=0.5)
    T = c2.number_input("Period T [s]", value=8.0)
    h1 = c3.number_input("Depth 1 [m]", value=7.0)
    h2 = c4.number_input("Depth 2 [m]", value=3.0)
    
    disp1 = solve_dispersion_relation(h1, T, g_in)
    disp2 = solve_dispersion_relation(h2, T, g_in)
    Ks = np.sqrt(disp1["Cg"] / disp2["Cg"])
    H2 = H1 * Ks
    
    render_abet_output(
        {"H1": f"{H1} m", "T": f"{T} s", "h1": f"{h1} m", "h2": f"{h2} m"},
        [f"L1 = {disp1['L']:.3f} m, C1 = {disp1['C']:.3f} m/s, Cg1 = {disp1['Cg']:.3f} m/s", f"L2 = {disp2['L']:.3f} m, C2 = {disp2['C']:.3f} m/s, Cg2 = {disp2['Cg']:.3f} m/s"],
        {"Shoaling Coeff Ks": f"{Ks:.3f}", "Shoaled Height H2": f"{H2:.3f} m"}
    )

elif selected_module == "M09: Refraction":
    st.markdown('<div class="main-title">Module 09: Shoaling & Refraction</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    H0 = c1.number_input("Deepwater H0 [m]", value=1.0)
    T = c2.number_input("Period T [s]", value=15.0)
    alpha0 = c3.number_input("Deep Angle α0 [deg]", value=75.0)
    h = c4.number_input("Local Depth h [m]", value=3.0)
    
    disp0 = solve_dispersion_relation(1000, T, g_in) # Deepwater proxy
    disp = solve_dispersion_relation(h, T, g_in)
    alpha_rad = np.arcsin((disp["C"] / disp0["C0"]) * np.sin(np.radians(alpha0)))
    alpha_deg = np.degrees(alpha_rad)
    Ks = np.sqrt((0.5 * disp0["C0"]) / disp["Cg"])
    Kr = np.sqrt(np.cos(np.radians(alpha0)) / np.cos(alpha_rad))
    H = H0 * Ks * Kr
    
    render_abet_output(
        {"H0": f"{H0} m", "T": f"{T} s", "alpha0": f"{alpha0}°", "h": f"{h} m"},
        [f"Cg0 = {disp0['Cg']:.2f} m/s", f"Cg_local = {disp['Cg']:.2f} m/s"],
        {"Refracted Angle α": f"{alpha_deg:.2f}°", "Shoaling Ks": f"{Ks:.3f}", "Refraction Kr": f"{Kr:.3f}", "Combined Height H": f"{H:.3f} m"}
    )

elif selected_module == "M10: Standing Waves":
    st.markdown('<div class="main-title">Module 10: Standing Wave Kinematics & Pressure</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    H = c1.number_input("Wave Height H [m]", value=1.5)
    T = c2.number_input("Period T [s]", value=5.0)
    h = c3.number_input("Depth h [m]", value=15.0)
    
    st.markdown("### Location & Time Coordinates")
    c4, c5, c6 = st.columns(3)
    disp = solve_dispersion_relation(h, T, g_in)
    L = disp['L']
    x_in = c4.text_input("Horizontal x [m] (e.g., L/3)", value="L/3")
    z_in = c5.text_input("Vertical z [m] (e.g., -h/3)", value="-h/3")
    t_in = c6.text_input("Time t [s] (e.g., 3*T/5)", value="3*T/5")
    
    x = eval(x_in.replace('L', str(L)))
    z = eval(z_in.replace('h', str(h)))
    t = eval(t_in.replace('T', str(T)))
    
    k = disp['k']
    sigma = disp['sigma']
    
    # Kinematics
    u = (H/2) * (g_in*k/sigma) * (np.cosh(k*(h+z))/np.cosh(k*h)) * np.sin(k*x) * np.sin(sigma*t)
    w = -(H/2) * (g_in*k/sigma) * (np.sinh(k*(h+z))/np.cosh(k*h)) * np.cos(k*x) * np.sin(sigma*t)
    ax = (H/2) * g_in*k * (np.cosh(k*(h+z))/np.cosh(k*h)) * np.sin(k*x) * np.cos(sigma*t)
    az = -(H/2) * g_in*k * (np.sinh(k*(h+z))/np.cosh(k*h)) * np.cos(k*x) * np.cos(sigma*t)
    
    # Trajectory & Pressure
    theta = np.degrees(np.arctan(-np.tanh(k*(h+z)) / np.tan(k*x))) if np.tan(k*x) != 0 else 90.0
    Ph = rho_in * g_in * (-z)
    Pd = rho_in * g_in * (H/2) * (np.cosh(k*(h+z))/np.cosh(k*h)) * np.cos(k*x) * np.cos(sigma*t)
    P_tot = Ph + Pd
    
    render_abet_output(
        {"H": f"{H} m", "T": f"{T} s", "h": f"{h} m", "x": f"{x:.2f} m", "z": f"{z:.2f} m", "t": f"{t:.2f} s"},
        [f"Wavelength L = {L:.3f} m", "Calculated using full standing wave partial derivatives"],
        {"Velocity u": f"{u:.3f} m/s", "Velocity w": f"{w:.3f} m/s", "Accel ax": f"{ax:.3f} m/s²", "Accel az": f"{az:.3f} m/s²", "Angle θ": f"{theta:.3f}°", "Hydro Pressure Ph": f"{Ph:.1f} N/m²", "Dynamic Pressure Pd": f"{Pd:.1f} N/m²", "Total Pressure P": f"{P_tot:.1f} N/m²"}
    )

elif selected_module == "M11: Diffraction":
    st.markdown('<div class="main-title">Module 11: Breakwater Diffraction</div>', unsafe_allow_html=True)
    scenario = st.radio("Scenario:", ("Semi-Infinite Breakwater (L11-P1)", "Breakwater Gap (L11-P2)"))
    
    if scenario == "Semi-Infinite Breakwater (L11-P1)":
        c1, c2, c3 = st.columns(3)
        Hi = c1.number_input("Incident Hi [m]", value=3.0)
        T = c2.number_input("T [s]", value=8.0)
        h = c3.number_input("Depth h [m]", value=15.0)
        c4, c5, c6 = st.columns(3)
        r = c4.number_input("Distance r [m]", value=40.9)
        theta = c5.number_input("Angle θ [deg]", value=60.0)
        alpha = c6.number_input("Incident Angle α [deg]", value=75.0)
        
        disp = solve_dispersion_relation(h, T, g_in)
        Kd = sommerfeld_Kd(r, np.radians(theta), np.radians(alpha), disp["k"])
        Hd = Hi * Kd
        
        render_abet_output(
            {"Hi": f"{Hi} m", "r": f"{r} m", "theta": f"{theta}°"},
            [f"Wavelength L = {disp['L']:.3f} m", f"r/L Ratio = {(r/disp['L']):.3f}"],
            {"Diffraction Coeff Kd": f"{Kd:.3f}", "Diffracted Height Hd": f"{Hd:.2f} m"}
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        Hi = c1.number_input("Incident Hi [m]", value=1.0)
        T = c2.number_input("T [s]", value=7.0)
        h = c3.number_input("Depth h [m]", value=10.0)
        B = c4.number_input("Gap Width B [m]", value=50.0)
        
        disp = solve_dispersion_relation(h, T, g_in)
        ratio = B / disp['L']
        Kd = 0.5 if ratio <= 1.0 else 1.0 # Centerline approximation for strong diffraction
        Hd = Hi * Kd
        diff_type = "Strong (Kd ≈ 0.5 on centerline)" if ratio <= 1.0 else "Weak"
        
        render_abet_output(
            {"Hi": f"{Hi} m", "B": f"{B} m", "h": f"{h} m"},
            [f"Wavelength L = {disp['L']:.3f} m", f"Ratio B/L = {ratio:.2f}", f"Diffraction Strength = {diff_type}"],
            {"Diffraction Coeff Kd": f"{Kd:.2f}", "Diffracted Height Hd": f"{Hd:.2f} m"}
        )
