# WTA (with inhibition) vs Baseline (no inhibition) — Clean Graph + Bars GIF (BIGGER + CLEAR)
# Output: wta_wta_vs_baseline_big.gif

import nest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

# -----------------------------
# Common parameters
# -----------------------------
T_total = 1000.0   # ms
T_half  = 500.0    # ms
RES     = 0.1      # ms kernel resolution

exc_model = "iaf_psc_delta"
inh_model = "iaf_psc_delta"
w_exc = 80.0       # pA (E->I and Poisson->E)
w_inh = -180.0     # pA (I->E)
d_syn = 1.5        # ms
dc_amp = 80.0      # pA

# Poisson rates (clear contrast)
rate_hi = 12000.0
rate_lo = 300.0

# Inst. rate window
ROLL_WIN = 80.0    # ms

# Animation settings
FRAMES = 160
FPS    = 24
times_frames = np.linspace(0.0, T_total, FRAMES)

# Colors for nodes (rate->color) and bars
cmap_exc = cm.get_cmap("Blues")
cmap_inh = cm.get_cmap("Purples")
MAX_RATE_FOR_COLOR = 150.0  # Hz saturation

def rate_to_color(rate_hz, cmap):
    x = np.clip(rate_hz / MAX_RATE_FOR_COLOR, 0.0, 1.0)
    x = 0.15 + 0.80 * x
    r, g, b, _ = cmap(x)
    return (r, g, b, 0.98)

def inst_rate(times, t, win_ms):
    if win_ms <= 0 or times.size == 0:
        return 0.0
    m = (times >= (t - win_ms)) & (times < t)
    return float(np.count_nonzero(m)) / (win_ms / 1000.0)  # Hz

# -----------------------------
# Simulation helpers
# -----------------------------
def simulate_wta():
    """Run WTA simulation (E1,E2,I; with inhibition) and return spike times."""
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": RES})

    E1 = nest.Create(exc_model, 1)
    E2 = nest.Create(exc_model, 1)
    I  = nest.Create(inh_model, 1)

    dc = nest.Create("dc_generator", 1, {"amplitude": dc_amp, "start": 0.0, "stop": T_total})
    for nc in [E1, E2, I]:
        nest.Connect(dc, nc)

    P1_hi = nest.Create("poisson_generator", 1, {"rate": rate_hi, "start": 0.0,    "stop": T_half})
    P2_lo = nest.Create("poisson_generator", 1, {"rate": rate_lo, "start": 0.0,    "stop": T_half})
    P1_lo = nest.Create("poisson_generator", 1, {"rate": rate_lo, "start": T_half, "stop": T_total})
    P2_hi = nest.Create("poisson_generator", 1, {"rate": rate_hi, "start": T_half, "stop": T_total})

    for src in [P1_hi, P1_lo]:
        nest.Connect(src, E1, syn_spec={"weight": w_exc, "delay": d_syn})
    for src in [P2_lo, P2_hi]:
        nest.Connect(src, E2, syn_spec={"weight": w_exc, "delay": d_syn})

    nest.Connect(E1, I, syn_spec={"weight": w_exc, "delay": d_syn})
    nest.Connect(E2, I, syn_spec={"weight": w_exc, "delay": d_syn})
    nest.Connect(I,  E1, syn_spec={"weight": w_inh, "delay": d_syn})
    nest.Connect(I,  E2, syn_spec={"weight": w_inh, "delay": d_syn})

    sr_E1 = nest.Create("spike_recorder")
    sr_E2 = nest.Create("spike_recorder")
    sr_I  = nest.Create("spike_recorder")
    nest.Connect(E1, sr_E1); nest.Connect(E2, sr_E2); nest.Connect(I, sr_I)

    nest.Simulate(T_total)

    t_E1 = np.array(nest.GetStatus(sr_E1, "events")[0]["times"])
    t_E2 = np.array(nest.GetStatus(sr_E2, "events")[0]["times"])
    t_I  = np.array(nest.GetStatus(sr_I,  "events")[0]["times"])
    return t_E1, t_E2, t_I

def simulate_baseline():
    """Run baseline simulation (E1,E2 only; no inhibition) and return spike times."""
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": RES})

    E1 = nest.Create(exc_model, 1)
    E2 = nest.Create(exc_model, 1)

    dc = nest.Create("dc_generator", 1, {"amplitude": dc_amp, "start": 0.0, "stop": T_total})
    nest.Connect(dc, E1); nest.Connect(dc, E2)

    P1_hi = nest.Create("poisson_generator", 1, {"rate": rate_hi, "start": 0.0,    "stop": T_half})
    P2_lo = nest.Create("poisson_generator", 1, {"rate": rate_lo, "start": 0.0,    "stop": T_half})
    P1_lo = nest.Create("poisson_generator", 1, {"rate": rate_lo, "start": T_half, "stop": T_total})
    P2_hi = nest.Create("poisson_generator", 1, {"rate": rate_hi, "start": T_half, "stop": T_total})

    for src in [P1_hi, P1_lo]:
        nest.Connect(src, E1, syn_spec={"weight": w_exc, "delay": d_syn})
    for src in [P2_lo, P2_hi]:
        nest.Connect(src, E2, syn_spec={"weight": w_exc, "delay": d_syn})

    sr_E1 = nest.Create("spike_recorder")
    sr_E2 = nest.Create("spike_recorder")
    nest.Connect(E1, sr_E1); nest.Connect(E2, sr_E2)

    nest.Simulate(T_total)

    t_E1 = np.array(nest.GetStatus(sr_E1, "events")[0]["times"])
    t_E2 = np.array(nest.GetStatus(sr_E2, "events")[0]["times"])
    return t_E1, t_E2

# -----------------------------
# Run simulations
# -----------------------------
tE1_w, tE2_w, tI_w = simulate_wta()
tE1_b, tE2_b       = simulate_baseline()

# Precompute bar y-limits (stable)
rE1_w_series = np.array([inst_rate(tE1_w, t, ROLL_WIN) for t in times_frames])
rE2_w_series = np.array([inst_rate(tE2_w, t, ROLL_WIN) for t in times_frames])
rE1_b_series = np.array([inst_rate(tE1_b, t, ROLL_WIN) for t in times_frames])
rE2_b_series = np.array([inst_rate(tE2_b, t, ROLL_WIN) for t in times_frames])
BAR_YMAX = max(1.0, rE1_w_series.max(), rE2_w_series.max(), rE1_b_series.max(), rE2_b_series.max()) * 1.2

# -----------------------------
# Figure & layout (BIGGER + CLEAR, extra spacing for legend)
# -----------------------------
plt.rcParams.update({"font.size": 14})

# Bigger canvas + constrained layout; add layout pads to increase spacing globally
fig = plt.figure(figsize=(12.8, 10.8), dpi=140, constrained_layout=True)
fig.suptitle("Winner-Take-All (WTA) — With Inhibition vs Baseline", fontsize=22, fontweight="bold", y=0.995)
fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.08, hspace=0.26)

# GridSpec with extra vertical room above the bars (for the legend)
gs = GridSpec(6, 1, height_ratios=[3.6, 0.18, 1.9, 0.70, 0.12, 0.0], figure=fig)
ax_graph = fig.add_subplot(gs[0, 0])
ax_bar   = fig.add_subplot(gs[2, 0])

# Graph geometry
pos = {"E1": (-1.8, -0.8), "E2": (1.8, -0.8), "I": (0.0, 1.2)}
R_NODE = 0.45

ax_graph.set_xlim(-3.1, 3.1); ax_graph.set_ylim(-2.0, 2.0); ax_graph.axis("off")

circle_E1 = Circle(pos["E1"], R_NODE, linewidth=2.4, edgecolor="black", facecolor=(0.8,0.85,1.0))
circle_E2 = Circle(pos["E2"], R_NODE, linewidth=2.4, edgecolor="black", facecolor=(0.8,0.85,1.0))
circle_I  = Circle(pos["I"],  R_NODE, linewidth=2.4, edgecolor="black", facecolor=(0.9,0.8,1.0))
for c in [circle_E1, circle_E2, circle_I]:
    ax_graph.add_patch(c)

def arrow(p_from, p_to, ls="-"):
    ax_graph.annotate(
        "", xy=p_to, xytext=p_from,
        arrowprops=dict(arrowstyle="-|>", lw=3.0, color="black", linestyle=ls,
                        shrinkA=18, shrinkB=18, mutation_scale=18)
    )
arrow(pos["E1"], pos["I"], ls="-")   # E->I (excitatory, solid)
arrow(pos["E2"], pos["I"], ls="-")
arrow(pos["I"],  pos["E1"], ls="--") # I->E (inhibitory, dashed)
arrow(pos["I"],  pos["E2"], ls="--")

# Labels inside nodes (name + Hz)
lbl_E1 = ax_graph.text(*pos["E1"], "E1\n0 Hz", ha="center", va="center",
                       fontsize=18, color="white", fontweight="bold")
lbl_E2 = ax_graph.text(*pos["E2"], "E2\n0 Hz", ha="center", va="center",
                       fontsize=18, color="white", fontweight="bold")
lbl_I  = ax_graph.text(*pos["I"],  "I\n0 Hz",  ha="center", va="center",
                       fontsize=18, color="white", fontweight="bold")

# ms counter (top-right, boxed)
counter_txt = ax_graph.text(0.985, 0.975, "t = 0 ms", transform=ax_graph.transAxes,
                            ha="right", va="top", fontsize=18,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))

# --- Bar chart: two bars per neuron (WTA vs Baseline) ---
ax_bar.set_ylim(0, BAR_YMAX)
centers = [0, 1]
bw = 0.36
x_wta  = [centers[0]-bw/2, centers[1]-bw/2]
x_base = [centers[0]+bw/2, centers[1]+bw/2]

bars_w = ax_bar.bar(x_wta,  [0, 0], width=bw, edgecolor="black", label="WTA (with inhibition)", color="#3b82f6")
bars_b = ax_bar.bar(x_base, [0, 0], width=bw, edgecolor="black", label="Baseline (no inhibition)", color="#9ca3af")

ax_bar.set_xticks(centers)
ax_bar.set_xticklabels(["E1", "E2"], fontsize=13)
ax_bar.set_ylabel("Inst. firing rate (Hz)", fontsize=13)
ax_bar.set_title("Excitation: WTA vs Baseline", fontsize=15, pad=18)  # extra pad
ax_bar.grid(axis="y", linestyle=":", alpha=0.4)

# Legend OUTSIDE the axes (well above bars, with extra padding)
ax_bar.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.48),   # higher to avoid overlap
    ncol=2,
    fontsize=13,
    borderaxespad=1.0
)

# Numeric labels above bars
label_w1 = ax_bar.text(x_wta[0],  0, "0 Hz", ha="center", va="bottom", fontsize=12)
label_w2 = ax_bar.text(x_wta[1],  0, "0 Hz", ha="center", va="bottom", fontsize=12)
label_b1 = ax_bar.text(x_base[0], 0, "0 Hz", ha="center", va="bottom", fontsize=12)
label_b2 = ax_bar.text(x_base[1], 0, "0 Hz", ha="center", va="bottom", fontsize=12)

# -----------------------------
# Animation update
# -----------------------------
def update(frame_idx):
    t = times_frames[frame_idx]

    # WTA instantaneous rates
    rE1_w = inst_rate(tE1_w, t, ROLL_WIN)
    rE2_w = inst_rate(tE2_w, t, ROLL_WIN)
    rI_w  = inst_rate(tI_w,  t, ROLL_WIN)

    # Baseline instantaneous rates
    rE1_b = inst_rate(tE1_b, t, ROLL_WIN)
    rE2_b = inst_rate(tE2_b, t, ROLL_WIN)

    # Update node colors (WTA)
    circle_E1.set_facecolor(rate_to_color(rE1_w, cmap_exc))
    circle_E2.set_facecolor(rate_to_color(rE2_w, cmap_exc))
    circle_I.set_facecolor(rate_to_color(rI_w,  cmap_inh))

    # Node labels with Hz
    lbl_E1.set_text(f"E1\n{rE1_w:.0f} Hz")
    lbl_E2.set_text(f"E2\n{rE2_w:.0f} Hz")
    lbl_I.set_text(f"I\n{rI_w:.0f} Hz")

    # Highlight winner (thicker border)
    if rE1_w >= rE2_w:
        circle_E1.set_linewidth(3.6); circle_E2.set_linewidth(2.4)
    else:
        circle_E2.set_linewidth(3.6); circle_E1.set_linewidth(2.4)

    # Update bars (WTA vs Baseline)
    bars_w[0].set_height(rE1_w); label_w1.set_position((x_wta[0], rE1_w)); label_w1.set_text(f"{rE1_w:.0f} Hz")
    bars_w[1].set_height(rE2_w); label_w2.set_position((x_wta[1], rE2_w)); label_w2.set_text(f"{rE2_w:.0f} Hz")
    bars_b[0].set_height(rE1_b); label_b1.set_position((x_base[0], rE1_b)); label_b1.set_text(f"{rE1_b:.0f} Hz")
    bars_b[1].set_height(rE2_b); label_b2.set_position((x_base[1], rE2_b)); label_b2.set_text(f"{rE2_b:.0f} Hz")

    # ms counter
    counter_txt.set_text(f"t = {int(t)} ms")

    return [circle_E1, circle_E2, circle_I, lbl_E1, lbl_E2, lbl_I,
            bars_w[0], bars_w[1], bars_b[0], bars_b[1],
            label_w1, label_w2, label_b1, label_b2, counter_txt]

ani = animation.FuncAnimation(fig, update, frames=len(times_frames), interval=1000/FPS, blit=False)

# Save GIF (bigger and sharper)
ani.save("wta_wta_vs_baseline_big.gif", writer=animation.PillowWriter(fps=FPS))
print("✅ Saved GIF to wta_wta_vs_baseline_big.gif")
