"""
Verstrooiing van twee massa's onder F = -gamma/r^2 (attractief).
Start met:  streamlit run scatter_app.py

Conventies (zoals in oefening 3):
  m1 = projectiel, m2 = target (in rust in LAB), q = m1/m2, p = impactparameter
  u  = beginsnelheid van m1 in LAB (op oneindig)
  psi = verstrooiingshoek in het ZM-frame, theta1/theta2 = verstrooiings-/terugstoothoek in LAB
"""
import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------------
# Fysica
# ----------------------------------------------------------------------------
def simulate(m1, m2, p, u, gamma, L=None):
    """Integreer beide deeltjes in het LAB-frame.

    Startpositie: m1 op (-L, y0), m2 in de oorsprong. Omdat de potentiaal 1/r
    traag naar nul gaat, corrigeren we de startsnelheid en y0 zodat u en p de
    waarden *op oneindig* zijn (energie- en impulsmomentbehoud van de relatieve beweging).
    """
    M = m1 + m2
    mu = (m1 * m2) / M
    r_c = gamma / (mu * u**2)                # karakteristieke lengteschaal
    if L is None:               #L is afstand tussen de massas bij begin van de simulatie en ook wanneer de simulatie stopt (als de afstand opnieuw L is)
        L = 400 * max(p, r_c)   #kies L groot genoeg

    # relatieve snelheid in het startpunt: 1/2 mu v^2 - gamma/L = 1/2 mu u^2
    v_rel = np.sqrt(u**2 + 2 * gamma / (mu * L))
    y0 = p * u / v_rel                       # zelfde impulsmoment: mu*v_rel*y0 = mu*u*p (p is gedef op inf y0 is op afstand L)
    # m2 in rust, m1 met snelheid v_rel  (LAB-frame, m2 start in rust)
    s0 = np.array([-L, y0, 0.0, 0.0, v_rel, 0.0, 0.0, 0.0]) #s is de state van het systeem (s0 is de begin condities) s = [x_1,y_1, x_2,y_2 , v_1x,v_1y , v_2x,v_2y]

    def rhs(t, s):
        #right hand side van de vergelijking: ds/dt = rhs(t,s)
        r1, r2, v1, v2 = s[0:2], s[2:4], s[4:6], s[6:8]
        d = r1 - r2
        r = np.hypot(*d)
        f = -gamma * d / r**3                # kracht op m1, hier gebruiken we d/r in plaats van eenheidsvector rhat
        return np.concatenate([v1, v2, f / m1, -f / m2])

    def verlaat(t, s):                       # relatieve afstand weer gelijk aan L
        return np.hypot(s[0] - s[2], s[1] - s[3]) - L
    verlaat.terminal = True
    verlaat.direction = 1

    t_max = 12 * L / u #als L te lang niet bereikt wordt dan is t_max bovengrens.
    sol = solve_ivp(rhs, (0, t_max), s0, method="DOP853", events=verlaat,
                    rtol=1e-10, atol=1e-12, dense_output=True)

    # verfijnde tijdsgrid voor een mooie curve, interpolatie van sol
    t = sol.t
    td = np.concatenate([np.linspace(a, b, 6)[:-1] for a, b in zip(t[:-1], t[1:])] + [t[-1:]])
    S = sol.sol(td)
    return dict(t=td, r1=S[0:2].T, r2=S[2:4].T, v1=S[4:6].T, v2=S[6:8].T,
                m1=m1, m2=m2, M=M, mu=mu, p=p, u=u, gamma=gamma, r_c=r_c, L=L)
    #returned een dict die we res noemen later (dit is al de info van de simulatie, res["v1"][0] is vb begin snelheid van massa 1)


def angle_between(a, b):
    return np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


def measured_angles(res):
    """Hoeken uit de numerieke eindsnelheden."""
    m1, m2, M = res["m1"], res["m2"], res["M"]
    v1i, v2i = res["v1"][0], res["v2"][0]
    v1f, v2f = res["v1"][-1], res["v2"][-1]
    V = (m1 * v1i + m2 * v2i) / M
    rel_i, rel_f = v1i - v2i, v1f - v2f
    psi = angle_between(rel_i, rel_f)                       # ZM-hoek
    theta1 = np.arctan2(abs(v1f[1]), v1f[0])                # tov. de x-as, 0..pi
    theta2 = np.arctan2(abs(v2f[1]), v2f[0])
    return dict(psi=psi, theta1=theta1, theta2=theta2, V=V)


def theory_angles(m1, m2, p, u, gamma):
    """Formules (3.32), (3.39), (3.40) uit de oplossingen."""
    q = m1 / m2
    E0 = 0.5 * m1 * u**2
    psi = 2 * np.arctan(gamma * (1 + q) / (2 * p * E0))     # (3.32)
    theta1 = np.arctan2(np.sin(psi), np.cos(psi) + q)       # (3.39), juiste tak
    theta2 = (np.pi - psi) / 2                              # (3.40)
    return dict(psi=psi, theta1=theta1, theta2=theta2)


# ----------------------------------------------------------------------------
# Streamlit-interface
# ----------------------------------------------------------------------------
def main():
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Verstrooiing m1–m2", layout="wide")
    st.title("Verstrooiing van twee massa's onder F = −γ/r²")

    with st.sidebar:
        st.header("Parameters")
        m1 = st.slider("m₁ (projectiel)", 0.1, 10.0, 1.0, 0.1)
        m2 = st.slider("m₂ (target)", 0.1, 10.0, 1.0, 0.1)
        p = st.slider("impactparameter p", 0.1, 10.0, 2.0, 0.1)
        u = st.slider("beginsnelheid u (LAB)", 0.2, 5.0, 1.0, 0.1)
        gamma = st.slider("γ", 0.1, 10.0, 1.0, 0.1)
        st.header("Weergave")
        frame = st.radio("Referentiestelsel", ["LAB", "ZM"], horizontal=True)
        window = st.slider("Venstergrootte", 2.0, 100.0, 15.0, 1.0)
        tfrac = st.slider("Tijd", 0.0, 1.0, 0.5, 0.01)

    res = simulate(m1, m2, p, u, gamma)
    meas, th = measured_angles(res), theory_angles(m1, m2, p, u, gamma)

    # --- banen in het gekozen frame ---
    r1, r2, t = res["r1"], res["r2"], res["t"]
    Rcm = (m1 * r1 + m2 * r2) / res["M"]
    if frame == "ZM":
        r1p, r2p, Rp = r1 - Rcm, r2 - Rcm, np.zeros_like(Rcm)
    else:
        r1p, r2p, Rp = r1, r2, Rcm

    # tijdsvenster: de tijden waarin m1 binnen het zichtbare venster zit
    inside = np.max(np.abs(r1), axis=1) < window
    idx = np.where(inside)[0]
    lo, hi = (idx[0], idx[-1]) if len(idx) > 1 else (0, len(t) - 1)
    k = int(lo + tfrac * (hi - lo))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(*r1p.T, color="tab:blue", label="m₁")    #massa 1 plotten (met dus r1p.T)
    ax.plot(*r2p.T, color="tab:red", label="m₂")    #massa 2
    ax.plot(*Rp.T, color="gray", ls=":", lw=1, label="zwaartepunt")     #zwaartepunt
    ax.plot(*r1p[k], "o", color="tab:blue", ms=5 + 3 * np.cbrt(m1))
    ax.plot(*r2p[k], "o", color="tab:red", ms=5 + 3 * np.cbrt(m2))
    ax.plot(*Rp[k], "x", color="k")
    ax.plot([r1p[k, 0], r2p[k, 0]], [r1p[k, 1], r2p[k, 1]], color="k", lw=0.5, alpha=0.5)
    ax.set_xlim(-window, window); ax.set_ylim(-window, window)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_title(f"{frame}-frame")
    c1, c2 = st.columns([3, 2])
    c1.pyplot(fig)
    plt.close(fig)

    # --- hoeken: numeriek vs. theorie ---
    deg = np.degrees
    c2.subheader("Hoeken (graden)")
    c2.table({
        "": ["ψ (ZM)", "θ₁ (LAB, projectiel)", "θ₂ (LAB, terugstoot)", "θ₁ + θ₂"],
        "numeriek": [deg(meas["psi"]), deg(meas["theta1"]), deg(meas["theta2"]),
                     deg(meas["theta1"] + meas["theta2"])],
        "formules": [deg(th["psi"]), deg(th["theta1"]), deg(th["theta2"]),
                     deg(th["theta1"] + th["theta2"])],
    })
    c2.caption(f"q = m₁/m₂ = {m1/m2:.3f}   |   μ = {res['mu']:.3f}   |   "
               f"V_CoM = {meas['V'][0]:.3f}   |   r_c = γ/(μu²) = {res['r_c']:.2f}")
    if abs(m1 - m2) < 1e-9:
        c2.info("m₁ = m₂: θ₁ + θ₂ = 90° (vergelijk met (3.46)–(3.47)).")


if __name__ == "__main__":
    main()