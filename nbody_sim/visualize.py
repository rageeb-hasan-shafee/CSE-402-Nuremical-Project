"""Interactive matplotlib viewer for the N-body solar-system simulation.

Controls:
  * Play/Pause button
  * "days/frame" slider - simulation speed (how many days advance per drawn frame)
  * "dt (days)" slider  - the velocity-Verlet integration time step
  * Reset button        - rebuild the solar system from its J2000 initial conditions
  * Body checkboxes      - toggle orbit trails on/off for individual bodies

Run directly:  python visualize.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation
from collections import deque

from orbital_elements import build_solar_system
from simulation import NBodySimulation, total_energy

TRAIL_LEN = 600
VIEW_LIMIT_AU = 32  # ~Neptune's orbit with margin


def main():
    names, masses, positions, velocities, colors = build_solar_system(include_moon=True)
    sim = NBodySimulation(names, masses, positions, velocities, colors)
    e0 = sim.energy()

    trails = {name: deque(maxlen=TRAIL_LEN) for name in names}
    visible = {name: True for name in names}

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_axes([0.06, 0.30, 0.68, 0.65])
    ax.set_aspect("equal")
    ax.set_xlim(-VIEW_LIMIT_AU, VIEW_LIMIT_AU)
    ax.set_ylim(-VIEW_LIMIT_AU, VIEW_LIMIT_AU)
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title("N-body Solar System — velocity Verlet integration")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("#1c1c1c")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors="#aaaaaa")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color("#eeeeee")

    sizes = np.clip(18 * (masses / masses.max()) ** 0.15, 4, 22)
    sizes[0] = 26  # Sun

    scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=sizes, zorder=5)
    trail_lines = {name: ax.plot([], [], lw=0.6, color=colors[i], alpha=0.6)[0]
                    for i, name in enumerate(names)}

    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white",
                         fontsize=9, va="top", family="monospace")

    # --- Controls -----------------------------------------------------
    ax_speed = fig.add_axes([0.12, 0.17, 0.55, 0.03])
    speed_slider = Slider(ax_speed, "days/frame", 0.5, 30.0, valinit=3.0)

    ax_dt = fig.add_axes([0.12, 0.11, 0.55, 0.03])
    dt_slider = Slider(ax_dt, "dt (days)", 0.02, 2.0, valinit=0.5)

    ax_play = fig.add_axes([0.78, 0.14, 0.09, 0.05])
    play_button = Button(ax_play, "Pause")

    ax_reset = fig.add_axes([0.78, 0.07, 0.09, 0.05])
    reset_button = Button(ax_reset, "Reset")

    ax_check = fig.add_axes([0.80, 0.30, 0.18, 0.55])
    ax_check.set_facecolor("#2a2a2a")
    check = CheckButtons(ax_check, names, [True] * len(names))
    for text in check.labels:
        text.set_color("white")
        text.set_fontsize(8)

    state = {"playing": True}

    def on_check(label):
        visible[label] = not visible[label]
        if not visible[label]:
            trails[label].clear()
            trail_lines[label].set_data([], [])

    check.on_clicked(on_check)

    def on_play(event):
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")

    play_button.on_clicked(on_play)

    def on_reset(event):
        nonlocal sim, e0
        _, m2, pos2, vel2, _ = build_solar_system(include_moon=True)
        sim = NBodySimulation(names, m2, pos2, vel2, colors)
        e0 = sim.energy()
        for name in names:
            trails[name].clear()
            trail_lines[name].set_data([], [])

    reset_button.on_clicked(on_reset)

    def update(frame):
        if state["playing"]:
            dt = dt_slider.val
            days_target = speed_slider.val
            n_sub = max(1, int(round(days_target / dt)))
            for _ in range(n_sub):
                sim.step(dt)

        scat.set_offsets(sim.positions[:, :2])

        for i, name in enumerate(names):
            if visible[name]:
                trails[name].append(sim.positions[i, :2].copy())
                pts = np.array(trails[name])
                trail_lines[name].set_data(pts[:, 0], pts[:, 1])

        e_now = sim.energy()
        drift = 100.0 * (e_now - e0) / abs(e0)
        time_text.set_text(
            f"t = {sim.t/365.25:6.2f} yr\n"
            f"energy drift = {drift:+.4f}%"
        )
        return [scat, time_text] + list(trail_lines.values())

    anim = FuncAnimation(fig, update, interval=30, blit=False)
    plt.show()
    return anim


if __name__ == "__main__":
    main()
