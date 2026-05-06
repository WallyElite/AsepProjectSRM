import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button

# ── Tune these ────────────────────────────────────────────────────────────────
Kp = 10.0
Ki =  0.1
Kd =  2.0
# ──────────────────────────────────────────────────────────────────────────────

# Spec parameters
m  = 0.5       # kg
g  = 9.81      # m/s²
h  = 0.15      # pivot-to-CoM (m)
I  = m * h**2  # moment of inertia (kg·m²)
dt = 0.01      # physics timestep (s)

MOTOR_CLAMP  = 5.0   # N·m
THETA_CLAMP  = 1.0   # rad — hard stop
L_VIS        = 1.0   # visual rod length (display only)
STEPS_PER_FRAME = 2  # physics steps per animation frame → ~50 fps display

state = {
    "theta":      0.1,
    "theta_dot":  0.0,
    "integral":   0.0,
    "prev_error": 0.0,
    "T_motor":    0.0,
}

def reset():
    state.update({"theta": 0.1, "theta_dot": 0.0,
                  "integral": 0.0, "prev_error": 0.0, "T_motor": 0.0})

def mess_up():
    sign = np.random.choice([-1, 1])
    state["theta"]     += sign * np.random.uniform(0.3, 0.7)
    state["theta_dot"] += sign * np.random.uniform(1.0, 3.0)

def pid(error):
    state["integral"] += error * dt
    derivative = (error - state["prev_error"]) / dt
    state["prev_error"] = error
    return np.clip(Kp * error + Ki * state["integral"] + Kd * derivative,
                   -MOTOR_CLAMP, MOTOR_CLAMP)

def step():
    T_grav          = m * g * h * np.sin(state["theta"])
    state["T_motor"] = pid(-state["theta"])
    theta_ddot       = (T_grav + state["T_motor"]) / I
    state["theta_dot"] += theta_ddot * dt
    state["theta"]     = np.clip(state["theta"] + state["theta_dot"] * dt,
                                 -THETA_CLAMP, THETA_CLAMP)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("#1a1a2e")

ax = fig.add_axes([0.04, 0.18, 0.60, 0.78])
ax.set_facecolor("#16213e")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Inverted Pendulum  —  motor at pivot", color="white", fontsize=12, pad=6)

motor_ring = patches.Circle((0, 0), 0.12, lw=2,
                             edgecolor="#00d4ff", facecolor="#0f3460", zorder=4)
ax.add_patch(motor_ring)
ax.plot(0, 0, "+", color="#00d4ff", markersize=10, markeredgewidth=1.5, zorder=5)

rod_line,  = ax.plot([], [], lw=5, color="#00d4ff", solid_capstyle="round", zorder=3)
bob_dot,   = ax.plot([], [], "o", color="#e94560", markersize=14, zorder=6)
torque_arc,= ax.plot([], [], lw=2.5, color="#ffa500", alpha=0.8, zorder=2)

angle_text  = ax.text(-1.55, 1.45, "", color="white",   fontsize=9, family="monospace")
torque_text = ax.text(-1.55, 1.30, "", color="#ffa500",  fontsize=9, family="monospace")
ax.text(-1.55, 1.15, f"Kp={Kp}  Ki={Ki}  Kd={Kd}",
        color="#888888", fontsize=8, family="monospace")

# ── Trace ─────────────────────────────────────────────────────────────────────
ax_tr = fig.add_axes([0.68, 0.18, 0.29, 0.78])
ax_tr.set_facecolor("#16213e")
ax_tr.set_xlim(0, 10)
ax_tr.set_ylim(-90, 90)
ax_tr.axhline(0, color="#e94560", lw=1, ls="--")
ax_tr.set_title("θ  (deg)", color="white", fontsize=10)
ax_tr.tick_params(colors="gray", labelsize=7)
ax_tr.set_xlabel("time (s)", color="gray", fontsize=8)
for sp in ax_tr.spines.values():
    sp.set_edgecolor("#0f3460")

trace_line, = ax_tr.plot([], [], color="#00d4ff", lw=1.5)
trace_t, trace_th = [], []

# ── Buttons ───────────────────────────────────────────────────────────────────
btn_reset = Button(fig.add_axes([0.10, 0.03, 0.16, 0.08]),
                   "Reset", color="#555577", hovercolor="#7777aa")
btn_mess  = Button(fig.add_axes([0.36, 0.03, 0.22, 0.08]),
                   "Mess Up!", color="#c0392b", hovercolor="#e74c3c")
for btn in (btn_reset, btn_mess):
    btn.label.set_color("white")
    btn.label.set_fontsize(11)

elapsed = [0.0]

btn_reset.on_clicked(lambda _: (reset(), trace_t.clear(),
                                trace_th.clear(), elapsed.__setitem__(0, 0.0)))
btn_mess.on_clicked(lambda _: mess_up())

# ── Animation ─────────────────────────────────────────────────────────────────
def animate(_):
    for _ in range(STEPS_PER_FRAME):
        step()
    elapsed[0] += dt * STEPS_PER_FRAME

    theta   = state["theta"]
    T_motor = state["T_motor"]

    rod_line.set_data([0, L_VIS * np.sin(theta)], [0, L_VIS * np.cos(theta)])
    bob_dot.set_data([L_VIS * np.sin(theta)], [L_VIS * np.cos(theta)])

    arc_span   = np.clip(T_motor / MOTOR_CLAMP, -1, 1) * 0.5
    arc_angles = np.linspace(np.pi / 2, np.pi / 2 + arc_span, 30)
    torque_arc.set_data(0.22 * np.cos(arc_angles), 0.22 * np.sin(arc_angles))

    angle_text.set_text(f"θ       = {np.degrees(theta):+.2f}°")
    torque_text.set_text(f"T_motor = {T_motor:+.2f} N·m")

    trace_t.append(elapsed[0])
    trace_th.append(np.degrees(theta))
    if len(trace_t) > 600:
        trace_t.pop(0)
        trace_th.pop(0)

    trace_line.set_data(trace_t, trace_th)
    ax_tr.set_xlim(max(0, elapsed[0] - 10), max(10, elapsed[0]))

    return rod_line, bob_dot, torque_arc, angle_text, torque_text, trace_line

ani = animation.FuncAnimation(
    fig, animate, interval=int(dt * STEPS_PER_FRAME * 1000),
    blit=True, cache_frame_data=False
)

plt.show()
