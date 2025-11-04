#!/usr/bin/env python3
"""
# Ski Jumper

This script simulates a simple ski jumper sliding along a slope.
It contains extensive English docstrings and line-by-line comments
intended for teaching and for students to study the code.

Rules:
See the README.md file in this repository.
https://github.com/s28041-pj/NAI/blob/main/Zad2/readme.md

Authors:
- Mikołaj Gurgul, Łukasz Aleksandrowicz

## Requirements
- Python 3.13
- numpy, matplotlib, PyQt5

## Installation (macOS)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python NAI_02.py

## How to run the game:

1. Install Python (>=3.13)
2. Clone this repository
3. pip install requirements.txt
4. Run the game:
    ```bash
    python NAI_02.py
    ```

"""

# Imports and backend configuration
import sys
import argparse

# Use Qt5Agg backend to enable interactive GUI windows on platforms
# where PyQt5 is available (macOS, Windows, Linux). If PyQt5 is missing,
# matplotlib will raise an error at import time when trying to use this backend.
import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# Simulation constants and tuning parameters
# Animation parameters
FPS = 30                      # frames per second for the animation
DT = 1.0 / FPS                # time step per frame (seconds)

# Viewport dimensions (world units)
VIEW_W, VIEW_H = 100.0, 60.0

# Skier geometry (width, height) used for rendering and collision checks
SKIER_W, SKIER_H = 4.0, 8.0
GROUND_CLEAR = SKIER_H / 2 + 1.5  # clearance from center to ground for collision tests

# Slope geometry: long straight-like slope defined by arrays
SLOPE_LENGTH = 3000.0
SLOPE_START_Y = 200.0
SLOPE_END_Y = -200.0

# How quickly the skier moves forward along the normalized slope parameter
BASE_SPEED = 0.12

# Camera following behavior (ratios and smoothing)
FOLLOW_X_RATIO, FOLLOW_Y_RATIO, CAM_SMOOTH = 0.25, 0.45, 0.12

# Conversion constant degrees -> radians
DEG = np.pi / 180.0

# Vertical physics for jump
GRAVITY = -30.0
JUMP_INIT_V = 18.0

# Wind and control tuning parameters (normal difficulty)
WIND_TORQUE_GAIN = 6.0 * DEG     # how much wind causes torque (radians per wind unit)
WIND_IMPULSE_GAIN = 0.06        # impulse factor for sudden wind changes

# Manual PID-like controller gains and limits
MAX_TORQUE_MAN = 80.0 * DEG
KP_MAN, KD_MAN = 12.0, 2.5

# Automatic controller (less aggressive)
MAX_TORQUE_AUTO = 35.0 * DEG
KP_AUTO, KD_AUTO = 6.0, 1.2

# Angular damping and falling threshold
ANG_DAMP = 1.2
ANGLE_FAIL_RAD = 35.0 * DEG     # if absolute angle exceeds this -> fall

# Slope construction (arrays) and helper to sample the slope by normalized t
slope_x = np.linspace(0, SLOPE_LENGTH, int(SLOPE_LENGTH / 2) + 1)
slope_y = np.linspace(SLOPE_START_Y, SLOPE_END_Y, slope_x.size)


def slope_point_at(t: float) -> tuple:
    """Return (x, y) coordinates along the slope for normalized parameter t in [0,1].

    The function linearly interpolates between discrete points stored in slope_x,
    slope_y. Using a normalized parameter makes forward motion simple (t increases).
    """
    # clamp to [0,1]
    idx = np.clip(t, 0.0, 1.0) * (slope_x.size - 1)
    i = int(min(idx, slope_x.size - 2))
    f = idx - i
    x = slope_x[i] + f * (slope_x[i + 1] - slope_x[i])
    y = slope_y[i] + f * (slope_y[i + 1] - slope_y[i])
    return x, y

# Matplotlib figure setup and HUD elements
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.08, bottom=0.24)
ax.set_aspect("equal")
ax.plot(slope_x, slope_y, color="saddlebrown", lw=3)
ax.set_title("Ski Jumper — NAI_02")

# Skier as a rounded rectangle patch (we rotate and translate this patch)
skier = patches.FancyBboxPatch((-SKIER_W / 2, -SKIER_H / 2),
                               SKIER_W, SKIER_H,
                               boxstyle="round,pad=0.3",
                               fc="royalblue", ec="black", lw=1.2)
ax.add_patch(skier)

# HUD texts shown on top-left of the axes using axis-fraction coordinates
mode_text = ax.text(0.02, 0.94, "Mode: Auto", transform=ax.transAxes)
status_text = ax.text(0.02, 0.88, "Status: OK", transform=ax.transAxes, color="green")
balance_text = ax.text(0.02, 0.82, "Balance: 0°", transform=ax.transAxes, fontsize=9)

# Simulation state variables (globals for pedagogical clarity)
# These globals store the dynamic state of the simulation. We use globals
# intentionally to keep the code straightforward for students.

t_pos = 0.0           # normalized progress along the slope (0..1)
angle = 0.0           # tilt angle in radians (positive = rotate CCW)
ang_vel = 0.0         # angular velocity (radians / second)
auto_mode = True      # whether automatic stabilizer is active
fallen = False        # flag indicating the skier has fallen
jump_y = 0.0          # vertical displacement when in the air (jump)
jump_v = 0.0          # vertical velocity while jumping
wind_speed = 0.0      # wind magnitude (slider 0..20)
wind_dir = 0.0        # wind direction factor (-1..1)
manual_target = 0.0   # desired angle in degrees when in manual mode
paused = False        # pause flag for animation
cam_left, cam_bottom = 0.0, slope_y.min()  # camera lower-left viewport coordinates
prev_signed_wind = 0.0  # previous signed wind value for computing impulse

# Widgets: sliders and a mode toggle button
axcolor = 'lightgoldenrodyellow'
s_ws = Slider(plt.axes([0.12, 0.14, 0.64, 0.035], facecolor=axcolor),
              'Wind Speed', 0, 20, valinit=0)
s_wd = Slider(plt.axes([0.12, 0.09, 0.64, 0.035], facecolor=axcolor),
              'Wind Dir (-1..1)', -1, 1, valinit=0)
s_man = Slider(plt.axes([0.12, 0.04, 0.64, 0.035], facecolor=axcolor),
               'Manual Balance (deg)', -40, 40, valinit=0)
b_mode = Button(plt.axes([0.8, 0.08, 0.16, 0.08]), 'Toggle Mode Auto', color='lightblue')

# Slider callbacks: set global variables when sliders change, but ignore input when fallen

def _set_ws(v):
    """Set wind speed from slider; ignore if skier has fallen."""
    global wind_speed
    if fallen:
        return
    wind_speed = float(v)


def _set_wd(v):
    """Set wind direction from slider; ignore if skier has fallen."""
    global wind_dir
    if fallen:
        return
    wind_dir = float(v)


def _set_man(v):
    """Set manual target angle (degrees) and switch to manual mode if needed."""
    global manual_target, auto_mode
    if fallen:
        return
    manual_target = float(v)
    if auto_mode:
        auto_mode = False
        b_mode.label.set_text("Toggle Mode Manual")

# connect callbacks
s_ws.on_changed(_set_ws)
s_wd.on_changed(_set_wd)
s_man.on_changed(_set_man)


def toggle_mode(event):
    """Toggle automatic/manual mode when the button is clicked.

    The event parameter is supplied by matplotlib's Button widget and is
    unused here, but we include it so this function can be passed directly
    to `on_clicked`.
    """
    global auto_mode
    if fallen:
        return
    auto_mode = not auto_mode
    b_mode.label.set_text("Toggle Mode Auto" if auto_mode else "Toggle Mode Manual")

b_mode.on_clicked(toggle_mode)

# Keyboard input handling
# We attach a key press event handler to the figure canvas. The handler updates
# global state (manual target, wind sliders, jump trigger, pause, reset).

def on_key(e):
    """Handle key press events from the user.

    Supported keys:
    - 'r' : reset simulation (always available)
    - left / right : nudge manual balance (and switch to manual)
    - up / down : increment/decrement wind speed
    - space : initiate jump if on ground
    - 'm' : toggle auto/manual mode
    - 'p' : pause / unpause animation
    """
    global manual_target, auto_mode, jump_y, jump_v, paused, fallen
    if not e.key:
        return
    k = e.key.lower()

    # reset always allowed
    if k == 'r':
        reset()
        return

    # when fallen, ignore other inputs (except reset)
    if fallen:
        return

    if k in ["left", "right"]:
        # adjust manual target by a small step and update slider visually
        step = -4 if k == "left" else 4
        manual_target = np.clip(manual_target + step, -40, 40)
        s_man.set_val(manual_target)
        if auto_mode:
            auto_mode = False
            b_mode.label.set_text("Toggle Mode Manual")
    elif k == "up":
        s_ws.set_val(min(wind_speed + 1, 20))
    elif k == "down":
        s_ws.set_val(max(wind_speed - 1, 0))
    elif k in [" ", "space"]:
        if not fallen and jump_y == 0:
            # set vertical velocity to jump initial
            jump_v = JUMP_INIT_V
    elif k == "m":
        toggle_mode(None)
    elif k == "p":
        paused = not paused

# bind key handler
fig.canvas.mpl_connect("key_press_event", on_key)

# Reset function: restore initial state and widgets

def reset():
    """Reset simulation globals and UI widgets to the initial state."""
    global t_pos, angle, ang_vel, fallen, jump_y, jump_v, prev_signed_wind
    global wind_speed, wind_dir, manual_target, paused, cam_left, cam_bottom

    t_pos = 0.0
    angle = 0.0
    ang_vel = 0.0
    fallen = False
    jump_y = 0.0
    jump_v = 0.0
    prev_signed_wind = 0.0
    wind_speed = 0.0
    wind_dir = 0.0
    manual_target = 0.0
    paused = False
    cam_left = 0.0
    cam_bottom = slope_y.min()

    # reset widget positions (this calls callbacks but callbacks respect 'fallen')
    s_ws.set_val(0.0)
    s_wd.set_val(0.0)
    s_man.set_val(0.0)
    status_text.set_text("Status: OK")
    status_text.set_color("green")
    b_mode.label.set_text("Toggle Mode Auto" if auto_mode else "Toggle Mode Manual")

# Collision helper functions

def skier_bottom_corners(x: float, y: float, angle: float, offset: float):
    """Return coordinates of the two bottom corners of the skier patch.

    The skier is modeled as a rectangle centered at (x,y) with width SKIER_W
    and height SKIER_H. We compute coordinates of the two bottom corners in
    world space after applying rotation by `angle` and vertical offset.

    Returns a numpy array shape (2,2) with rows [x1,y1], [x2,y2].
    """
    # rotation matrix for angle
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    half_w, half_h = SKIER_W / 2, SKIER_H / 2
    # local coordinates of two bottom corners (left-bottom, right-bottom)
    local = np.array([[-half_w, -half_h], [half_w, -half_h]])
    pts = local @ rot.T
    pts[:, 0] += x
    pts[:, 1] += y + offset
    return pts


def below_slope(xc: float, yc: float) -> bool:
    """Return True if point (xc,yc) is below or on the slope surface.

    The slope is approximated as a linear connection between start and end Y
    across x in [0, SLOPE_LENGTH]. If xc is outside that range, function
    returns False (no collision with slope body).
    """
    if xc < 0 or xc > SLOPE_LENGTH:
        return False
    frac = xc / SLOPE_LENGTH
    slope_y_line = SLOPE_START_Y + (SLOPE_END_Y - SLOPE_START_Y) * frac
    return yc <= slope_y_line

# Animation initialization and update functions

def init():
    """Initial placement of the skier and initial viewport limits.

    Called once by FuncAnimation to set up artists before the animation loop.
    """
    x, y = slope_point_at(0.0)
    # place skier slightly above the slope using GROUND_CLEAR
    skier.set_transform(transforms.Affine2D().translate(x, y + GROUND_CLEAR) + ax.transData)
    # set initial camera window
    ax.set_xlim(x - VIEW_W * 0.1, x + VIEW_W * 0.9)
    ax.set_ylim(y - VIEW_H * 0.45, y + VIEW_H * 0.55)
    return (skier,)


def animate(frame):
    """Update function called every frame by FuncAnimation.

    Steps performed here:
    1. Advance forward progress t_pos unless paused or fallen
    2. Integrate jump vertical position if in the air
    3. Compute wind torque and impulse from changes in wind
    4. Compute controller torque (auto or manual)
    5. Integrate angular velocity and angle
    6. Check collisions and angle-based falls
    7. Update skier transform, camera and HUD

    Returns a tuple of artists that were modified (required by FuncAnimation).
    """
    global t_pos, angle, ang_vel, fallen, jump_y, jump_v, cam_left, cam_bottom
    global prev_signed_wind, wind_speed, wind_dir

    if paused:
        return (skier,)

    # if fallen we freeze simulation state and only update status text
    if fallen:
        status_text.set_text("Status: FALLEN! Press R to restart")
        status_text.set_color("red")
        return (skier,)

    # advance along slope (normalize increment by slope length to keep BASE_SPEED meaningful)
    t_pos = min(t_pos + (BASE_SPEED * DT) / (SLOPE_LENGTH / 1000.0), 1.0)
    x, y = slope_point_at(t_pos)

    # jump: simple vertical kinematics while in air
    if jump_y > 0 or jump_v > 0:
        jump_y += jump_v * DT
        jump_v += GRAVITY * DT
        if jump_y < 0:
            # landed (stabilize)
            jump_y = 0
            jump_v = 0

    # wind influence
    signed_wind = wind_speed * wind_dir  # negative or positive depending on direction
    forward_difficulty = 1.0 + 0.7 * (BASE_SPEED / 0.12)
    wind_torque = signed_wind * WIND_TORQUE_GAIN * forward_difficulty

    # impulse due to abrupt wind changes — compute delta and convert to torque impulse
    wind_change = signed_wind - prev_signed_wind
    prev_signed_wind = signed_wind
    # dividing by DT makes the impulse larger for small DT, approximating instantaneous change
    impulse_torque = wind_change * WIND_IMPULSE_GAIN / DT

    # controller: compute control torque depending on mode
    if auto_mode:
        target = 0.0
        ctrl = np.clip(KP_AUTO * (target - angle) - KD_AUTO * ang_vel, -MAX_TORQUE_AUTO, MAX_TORQUE_AUTO)
    else:
        target = manual_target * DEG
        ctrl = np.clip(KP_MAN * (target - angle) - KD_MAN * ang_vel, -MAX_TORQUE_MAN, MAX_TORQUE_MAN)

    # total torque on the skier (wind + impulse + controller - damping)
    torque = wind_torque + impulse_torque + ctrl - ANG_DAMP * ang_vel

    # integrate angular motion
    ang_vel += torque * DT
    angle += ang_vel * DT

    # collision checks using bottom corners of the rotated rectangle
    corners = skier_bottom_corners(x, y, angle, GROUND_CLEAR + jump_y)
    for cx, cy in corners:
        if below_slope(cx, cy):
            fallen = True
            status_text.set_text("Status: FALLEN! (touch) Press R to restart")
            status_text.set_color("red")
            break

    # if tilt exceeds safe threshold -> fall
    if not fallen and abs(angle) > ANGLE_FAIL_RAD:
        fallen = True
        status_text.set_text("Status: FALLEN! (angle) Press R to restart")
        status_text.set_color("red")

    # rendering: set patch transform (rotation then translation)
    rot_deg = angle / DEG
    tform = transforms.Affine2D().rotate_deg_around(0, 0, rot_deg).translate(x, y + GROUND_CLEAR + jump_y)
    skier.set_transform(tform + ax.transData)

    # camera smooth follow
    cam_left += ((x - FOLLOW_X_RATIO * VIEW_W) - cam_left) * CAM_SMOOTH
    cam_bottom += ((y - FOLLOW_Y_RATIO * VIEW_H) - cam_bottom) * CAM_SMOOTH
    ax.set_xlim(cam_left, cam_left + VIEW_W)
    ax.set_ylim(cam_bottom, cam_bottom + VIEW_H)

    # HUD updates
    balance_text.set_text(f"Balance: {rot_deg:.1f}°")
    mode_text.set_text("Mode: Auto" if auto_mode else "Mode: Manual")
    if not fallen:
        status_text.set_text("Status: OK")
        status_text.set_color("green")

    return (skier,)

# create animation object (very many frames so the simulation appears continuous)
anim = FuncAnimation(fig, animate, init_func=init, frames=200000, interval=1000 / FPS, blit=False)

# small help text shown below the plot
fig.text(0.5, 0.01,
         "←/→ balance | ↑/↓ wind | Space jump | M mode | P pause | R reset",
         ha='center', fontsize=9)

# Export helper: writes requirements.txt and README.md to disk

def export_files():
    """Write REQUIREMENTS_TXT and README_MD to files in the current directory."""
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(REQUIREMENTS_TXT.strip() + '')
    with open('readme.md', 'w', encoding='utf-8') as f:
        f.write(README_MD.strip() + '')
    print("Wrote: requirements.txt, README.md")

# Main entry point: command line parsing and running the GUI

def main(argv=None):
    """Main function: parse arguments and either export helpers or show the plot.

    Use `--export` to write requirements.txt and README.md and exit.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Ski Jumper NAI_02')
    parser.add_argument('--export', action='store_true', help='Write requirements.txt and README.md and exit')
    args = parser.parse_args(argv)

    if args.export:
        export_files()
        return

    # Display the interactive matplotlib window. If PyQt5 or compatible backend is
    # unavailable this call will raise; install PyQt5 via pip if needed.
    plt.show()


if __name__ == '__main__':
    main()
