""" Jones vector """
import numpy as np
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d
from fractions import Fraction


""" Global variables """
e_0x = 1.
e_0y = 0.
phi_x_pi = 0.
phi_y_pi = 0.

k = 1.
omega = 1.
t = 0

is_norm = False
scale_norm = 1.

num_arrows = 40
arrows = []


""" Animation control """
is_play = False

""" Axis vectors """

""" Create figure and axes """
title_tk = "Jones vector "
title_ax0 = title_tk

x_min = 0.
x_max = 4.
y_min = -2.
y_max = 2.
z_min = -2.
z_max = 2.

# fig = Figure(facecolor='black')
fig = Figure(constrained_layout=True)
ax0 = fig.add_subplot(121, projection="3d")
ax0.set_box_aspect((2, 1, 1))
ax0.grid()
ax0.set_title(title_ax0, color="black")
ax0.set_xlabel(r"$z (\pi)$")
ax0.set_ylabel("x (horizontal)")
ax0.set_zlabel("y (vertical)")
ax0.set_xlim(x_min, x_max)
ax0.set_ylim(y_min, y_max)
ax0.set_zlim(z_min, z_max)

# ax0.set_facecolor("black")
# ax0.axis('off')

ax1 = fig.add_subplot(122)
# ax1.set_title("***")
# ax1.set_xlabel("***")
# ax1.set_ylabel("***")
ax1.set_xlim(0, 99)
ax1.set_ylim(0, 99)
ax1.invert_yaxis()

ax1.set_aspect("equal")
# ax1.set_aspect(30)
# ax1.grid()
# ax1.set_xticks(np.arange(0, 360, 60))
ax1.axis('off')


""" Embed in Tkinter """


def delayed_resize():
    width_new, height_new = canvas.get_width_height()
    # print("Delayed resize:", width_new, height_new)
    new_font_size = int(font_size * height_new / height_init)
    # print(font_size, new_font_size)

    txt_e.set_fontsize(new_font_size)
    txt_element_ex.set_fontsize(new_font_size)
    txt_element_ey.set_fontsize(new_font_size)

    txt_equal.set_fontsize(new_font_size)
    txt_element_ex1.set_fontsize(new_font_size)
    txt_element_ey1.set_fontsize(new_font_size)

    txt_equal1.set_fontsize(new_font_size)
    txt_element_ex2.set_fontsize(new_font_size)
    txt_element_ey2.set_fontsize(new_font_size)

    txt_norm.set_fontsize(new_font_size * 0.8)

    canvas.draw_idle()


root = tk.Tk()
root.title(title_tk)
canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack(expand=True, fill="both")
toolbar = NavigationToolbar2Tk(canvas, root)
root.bind("<Configure>", lambda e: root.after(100, delayed_resize))


""" Global objects of Tkinter """
var_e_0x = tk.StringVar(root)
var_e_0y = tk.StringVar(root)
var_phi_x_pi = tk.StringVar(root)
var_phi_y_pi = tk.StringVar(root)
var_is_norm = tk.IntVar(root)

""" Classes and functions """


class Counter:
    def __init__(self, is3d=None, ax=None, xy=None, z=None, label="", color=None):
        self.is3d = is3d if is3d is not None else False
        self.ax = ax
        self.x, self.y = xy[0], xy[1]
        self.z = z if z is not None else 0
        self.label = label
        self.color  = color

        self.count = 0

        if not is3d:
            self.txt_step = self.ax.text(self.x, self.y, self.label + str(self.count), color=color)
        else:
            self.txt_step = self.ax.text2D(self.x, self.y, self.label + str(self.count), color=color)
            self.xz, self.yz, _ = proj3d.proj_transform(self.x, self.y, self.z, self.ax.get_proj())
            self.txt_step.set_position((self.xz, self.yz))

    def count_up(self):
        self.count += 1
        self.txt_step.set_text(self.label + str(self.count))

    def reset(self):
        self.count = 0
        self.txt_step.set_text(self.label + str(self.count))

    def get(self):
        return self.count


class Arrow3d:
    def __init__(self, ax, x, y, z, u, v, w, color, line_width, line_style, arrow_length_ratio):
        self.ax = ax
        self.x, self.y, self.z = x, y, z
        self.u, self.v, self.w = u, v, w
        self.color = color
        self.line_width = line_width
        self.line_style = line_style
        self.arrow_length_ratio = arrow_length_ratio

        self.qvr = self.ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w,
                                  length=1, color=self.color, normalize=False,
                                  linewidth=self.line_width, linestyle=self.line_style,
                                  arrow_length_ratio=self.arrow_length_ratio)

    def _update_quiver(self):
        self.qvr.remove()
        self.qvr = self.ax.quiver(self.x, self.y, self.z, self.u, self.v, self.w,
                                  length=1, color=self.color, normalize=False,
                                  linewidth=self.line_width, linestyle=self.line_style,
                                  arrow_length_ratio=self.arrow_length_ratio)

    def set_vector(self, u, v, w):
        self.u, self.v, self.w = u, v, w
        self._update_quiver()

    def get_vector(self):
        return np.array([self.u, self.v, self.w])


def complex_exp_latex(n, m):
    # Case n = 0 → always 0
    if abs(n) < 1e-12:
        return "0"

    frac = Fraction(m).limit_denominator(12)  # Convert decimal m to fraction
    numerator, denominator = frac.numerator, frac.denominator

    # Special angle values (period 2π)
    angle = (m % 2)
    if angle == 0:
        base = 1
    elif angle == 1:
        base = -1
    elif angle == 0.5:
        base = 1j
    elif angle == 1.5:
        base = -1j
    else:
        # General exponential form
        if denominator == 1:
            if n == 1:
                return rf"e^{{i {numerator}\pi}}"
            elif n == -1:
                return rf"-e^{{i {numerator}\pi}}"
            else:
                return rf"{n}e^{{i {numerator}\pi}}"
        else:
            if n == 1:
                return rf"e^{{i \frac{{{numerator}}}{{{denominator}}}\pi}}"
            elif n == -1:
                return rf"-e^{{i \frac{{{numerator}}}{{{denominator}}}\pi}}"
            else:
                return rf"{n}e^{{i \frac{{{numerator}}}{{{denominator}}}\pi}}"

    # Multiply n with special values {1, -1, i, -i}
    result = n * base

    # Convert result to LaTeX string
    if abs(result - 1) < 1e-12:
        return "1"
    elif abs(result + 1) < 1e-12:
        return "-1"
    elif abs(result - 1j) < 1e-12:
        return "i"
    elif abs(result + 1j) < 1e-12:
        return "-i"
    elif isinstance(result, complex):
        re, im = result.real, result.imag
        if abs(re) < 1e-12:  # Pure imaginary
            if abs(im - 1) < 1e-12:
                return "i"
            elif abs(im + 1) < 1e-12:
                return "-i"
            else:
                return rf"{im}i"
        elif abs(im) < 1e-12:  # Pure real
            return str(re)
        else:  # General complex number
            re_str = str(re)
            if abs(im - 1) < 1e-12:
                im_str = "+i"
            elif abs(im + 1) < 1e-12:
                im_str = "-i"
            else:
                im_str = f"{'+' if im > 0 else ''}{im}i"
            return re_str + im_str
    else:
        return str(result)


def update_diagram():
    global wave_e_0x, wave_e_0y
    global plt_wave_e_0x, plt_wave_e_0y, plt_wave_e

    t = cnt.get() / 10.
    wave_e_0x = scale_norm * e_0x * np.cos((k * x - omega * t + phi_x_pi) * np.pi)
    wave_e_0y = scale_norm * e_0y * np.cos((k * x - omega * t + phi_y_pi) * np.pi)

    plt_wave_e_0x.set_data_3d(x, wave_e_0x, z0)
    plt_wave_e_0y.set_data_3d(x, y0, wave_e_0y)
    plt_wave_e.set_data_3d(x, wave_e_0x, wave_e_0y)

    for i in range(num_arrows):
        x_ = x_min + i * (x_max - x_min) / num_arrows
        y_ = scale_norm * e_0x * np.cos((k * x_ - omega * t + phi_x_pi) * np.pi)
        z_ = scale_norm * e_0y * np.cos((k * x_ - omega * t + phi_y_pi) * np.pi)
        arrows[i].set_vector(0., y_, z_)


def update_text():
    global element_ex1, element_ex2, element_ey1, element_ey2
    element_ex1 = fr"${e_0x} e^{{i{phi_x_pi}\pi}}$"
    txt_element_ex1.set_text(element_ex1)

    element_ey1 = fr"${e_0y} e^{{i{phi_y_pi}\pi}}$"
    txt_element_ey1.set_text(element_ey1)

    element_ex2 = f"${complex_exp_latex(e_0x, phi_x_pi)}$"
    element_ey2 = f"${complex_exp_latex(e_0y, phi_y_pi)}$"

    txt_element_ex2.set_text(element_ex2)
    txt_element_ey2.set_text(element_ey2)

    if scale_norm == 1:
        txt_norm.set_text("")
    else:
        if abs(e_0x) == abs(e_0y):
            txt_norm.set_text(f"$\\frac{{1}}{{\\sqrt{{2}}}}$")
        else:
            txt_norm.set_text(str(round(scale_norm, 2)))


def get_scale_norm():
    if is_norm:
        r = np.sqrt(e_0x ** 2. + e_0y ** 2.)
        if r != 0:
            scale = 1. / r
        else:
            scale = 1.
    else:
        scale = 1.

    return scale


def set_is_norm(value):
    global is_norm, scale_norm
    is_norm = value
    scale_norm = get_scale_norm()
    update_diagram()
    update_text()


def set_e_0x(value):
    global e_0x, element_ex1, scale_norm
    e_0x = value
    scale_norm = get_scale_norm()
    update_text()
    update_diagram()


def set_e_0y(value):
    global e_0y, element_ey1, scale_norm
    e_0y = value
    scale_norm = get_scale_norm()
    update_text()
    update_diagram()


def set_phi_x_pi(value):
    global phi_x_pi, element_ex1
    phi_x_pi = value
    update_text()
    update_diagram()


def set_phi_y_pi(value):
    global phi_y_pi, element_ey1
    phi_y_pi = value
    update_text()
    update_diagram()


def create_parameter_setter():
    # Horizontal
    frm_e_0x = ttk.Labelframe(root, relief="ridge", text="x (horizontal)", labelanchor="n")
    frm_e_0x.pack(side='left', fill=tk.Y)

    lbl_e_0x = tk.Label(frm_e_0x, text="E_{0x}")
    lbl_e_0x.pack(side='left')

    # var_e_0x = tk.StringVar(root)
    var_e_0x.set(str(e_0x))
    spn_e_0x = tk.Spinbox(
        frm_e_0x, textvariable=var_e_0x, format="%.1f", from_=-10., to=10., increment=0.1,
        command=lambda: set_e_0x(float(var_e_0x.get())), width=5
    )
    spn_e_0x.pack(side="left")

    lbl_phi_x = tk.Label(frm_e_0x, text="Phi_x(*pi)")
    lbl_phi_x.pack(side='left')

    # var_phi_x_pi = tk.StringVar(root)
    var_phi_x_pi.set(str(phi_x_pi))
    spn_phi_x = tk.Spinbox(
        frm_e_0x, textvariable=var_phi_x_pi, format="%.2f", from_=-4., to=4., increment=0.05,
        command=lambda: set_phi_x_pi(float(var_phi_x_pi.get())), width=5
    )
    spn_phi_x.pack(side="left")

    # Vertical
    frm_e_0y = ttk.Labelframe(root, relief="ridge", text="y (vertical)", labelanchor="n")
    frm_e_0y.pack(side='left', fill=tk.Y)

    lbl_e_0y = tk.Label(frm_e_0y, text="E_{0y}")
    lbl_e_0y.pack(side='left')

    # var_e_0y = tk.StringVar(root)
    var_e_0y.set(str(e_0y))
    spn_e_0y = tk.Spinbox(
        frm_e_0y, textvariable=var_e_0y, format="%.1f", from_=-10., to=10., increment=0.1,
        command=lambda: set_e_0y(float(var_e_0y.get())), width=5
    )
    spn_e_0y.pack(side="left")

    lbl_phi_y = tk.Label(frm_e_0y, text="Phi_y(*pi)")
    lbl_phi_y.pack(side='left')

    # var_phi_y_pi = tk.StringVar(root)
    var_phi_y_pi.set(str(phi_x_pi))
    spn_phi_y = tk.Spinbox(
        frm_e_0y, textvariable=var_phi_y_pi, format="%.2f", from_=-4., to=4., increment=0.05,
        command=lambda: set_phi_y_pi(float(var_phi_y_pi.get())), width=5
    )
    spn_phi_y.pack(side="left")

    # Normalize
    frm_norm = ttk.Labelframe(root, relief="ridge", text="Normalize", labelanchor="n")
    frm_norm.pack(side='left', fill=tk.Y)

    # var_is_norm = tk.IntVar(root)
    chk_is_norm = tk.Checkbutton(frm_norm, text="Apply", variable=var_is_norm,
                                 command=lambda: set_is_norm(var_is_norm.get()))
    chk_is_norm.pack(side='left')
    var_is_norm.set(is_norm)


def create_animation_control():
    frm_anim = ttk.Labelframe(root, relief="ridge", text="Animation; apply e^i(kz-omega*t)", labelanchor="n")
    frm_anim.pack(side="left", fill=tk.Y)
    btn_play = tk.Button(frm_anim, text="Play/Pause", command=switch)
    btn_play.pack(side="left")
    btn_reset = tk.Button(frm_anim, text="Reset", command=reset)
    btn_reset.pack(side="left")
    # btn_clear = tk.Button(frm_anim, text="Clear path", command=lambda: aaa())
    # btn_clear.pack(side="left")


def create_center_lines():
    ln_axis_x = art3d.Line3D([x_min, x_max], [0., 0.], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(ln_axis_x)
    ln_axis_y = art3d.Line3D([0., 0.], [y_min, y_max], [0., 0.], color="gray", ls="-.", linewidth=1)
    ax0.add_line(ln_axis_y)
    ln_axis_z = art3d.Line3D([0., 0.], [0., 0.], [z_min, z_max], color="gray", ls="-.", linewidth=1)
    ax0.add_line(ln_axis_z)


def draw_static_diagrams():
    create_center_lines()
    c = Circle((0, 0), 1, ec='gray', fill=False)
    ax0.add_patch(c)
    art3d.pathpatch_2d_to_3d(c, z=0, zdir="x")

    for i in range(4):
        ln_aux = art3d.Line3D([0, 0.],
                              [0., np.cos(i * np.pi / 2. + np.pi / 4.)],
                              [0., np.sin(i * np.pi / 2. + np.pi / 4.)], color="gray", ls="-.", linewidth=1)
        ax0.add_line(ln_aux)

    ax0.text(0, 1, 0, r"$\mathbf{H}$", color="black", va='center')
    ax0.text(0, - 1, 0, r"$\mathbf{H}$", color="black", va='center')
    ax0.text(0, 0, 1, r"$\mathbf{V}$", color="black", va='center')
    ax0.text(0, 0, - 1, r"$\mathbf{V}$", color="black", va='center')
    ax0.text(0, 0.7, 0.7, r"$\mathbf{D}$", color="black", va='center')
    ax0.text(0, - 0.7, - 0.7, r"$\mathbf{D}$", color="black", va='center')
    ax0.text(0, 0.7, - 0.7, r"$\mathbf{A}$", color="black", va='center')
    ax0.text(0, - 0.7, 0.7, r"$\mathbf{A}$", color="black", va='center')

    theta1 = np.linspace(0, np.pi * 3 / 2, 100)
    r = 0.5
    y = r * np.cos(theta1)
    z = r * np.sin(theta1)
    x = np.zeros_like(theta1) + x_max - 0.2

    ax0.plot(x, y, z, color="darkgray")

    ax0.quiver(x_max - 0.2, 0, - 0.5, 0, 0.1, 0, length=1.5, arrow_length_ratio=1.5, color="darkgray")

    theta2 = np.linspace(- np.pi / 2, np.pi, 100)
    r = 0.5
    y = r * np.cos(theta2)
    z = r * np.sin(theta2)
    x = np.zeros_like(theta2) + x_max + 0.2

    ax0.plot(x, y, z, color="darkgray")

    ax0.quiver(x_max + 0.2, 0, - 0.5, 0, - 0.1, 0, length=1.5, arrow_length_ratio=1.5, color="darkgray")

    ax0.text(x_max + 0.2, 0, 0.5, r"$\mathbf{R}$", color="black", va='center')
    ax0.text(x_max - 0.2, 0, 0.5, r"$\mathbf{L}$", color="black", va='center')


def draw_bracket(ax, x, y, width, height, color):
    y_upper, y_lower = y + height / 2, y - height / 2
    arm = height * 0.1

    bracket_left_x = np.array([x + arm, x, x, x + arm])
    bracket_left_y = np.array([y_upper, y_upper, y_lower, y_lower])
    plt_bracket_left = ax.plot(bracket_left_x, bracket_left_y, c=color)

    bracket_right_x = np.array([x - arm + width, x + width, x + width, x - arm + width])
    bracket_right_y = np.array([y_upper, y_upper, y_lower, y_lower])
    plt_bracket_right = ax.plot(bracket_right_x, bracket_right_y, c=color)


def reset():
    global is_play
    if is_play:
        is_play = not is_play
    cnt.reset()
    update_diagram()


def switch():
    global is_play
    if is_play:
        is_play = False
    else:
        is_play = True


def update(f):
    if is_play:
        cnt.count_up()
        update_diagram()


""" main loop """
if __name__ == "__main__":
    cnt = Counter(ax=ax0, is3d=True, xy=np.array([x_min, y_max]), z=z_max, label="t(/10)=", color="black")
    draw_static_diagrams()
    create_animation_control()
    create_parameter_setter()
    font_size = 16

    vector_e = r"$\mathbf{J} =$"
    element_ex = r"$E_{0x} e^{i\phi_x}$"
    element_ey = r"$E_{0y} e^{i\phi_y}$"
    ef_wave_0x = (r"$E_x = E_{0x} e^{i(kz - \omega t +\phi_x)} = E_{0x} e^{i(kz- \omega t)} e^{i\phi_x} = "
                 r"E_{0x} e^{i\phi_x} e^{i(kz - \omega t)} $")
    ef_wave_0y = (r"$E_y = E_{0y} e^{i(kz - \omega t +\phi_y)} = E_{0y} e^{i(kz- \omega t)} e^{i\phi_y} = "
                 r"E_{0y} e^{i\phi_y} e^{i(kz - \omega t)} $")
    # euler = r"$e^{i\theta} = \cos \theta + i \sin \theta$"

    equal = r"$=$"

    element_ex1 = fr"${e_0x} e^{{i{phi_x_pi}\pi}}$"
    element_ey1 = fr"${e_0y} e^{{i{phi_y_pi}\pi}}$"

    txt_e = ax1.text(5, 20, vector_e, fontsize=font_size, va='center')
    txt_element_ex = ax1.text(22, 15, element_ex, fontsize=font_size, va='center')
    txt_element_ey = ax1.text(22, 25, element_ey, fontsize=font_size, va='center')

    txt_equal = ax1.text(52, 20, equal, fontsize=font_size, va='center')
    txt_element_ex1 = ax1.text(62, 15, element_ex1, fontsize=font_size, va='center')
    txt_element_ey1 = ax1.text(62, 25, element_ey1, fontsize=font_size, va='center')

    txt_note = ax1.text(5, 70, "Electric field wave", fontsize=font_size * 0.8, va='center')
    txt_wave_0x = ax1.text(10, 76, ef_wave_0x, fontsize=font_size * 0.8, va='center', color="blue")
    txt_wave_0y = ax1.text(10, 82, ef_wave_0y, fontsize=font_size * 0.8, va='center', color="green")
    txt_note1 = ax1.text(10, 90, "Note; the graph only represents the real part.",
                         fontsize=font_size * 0.6, va='center')
    # txt_euler = ax1.text(10, 96, euler, fontsize=font_size * 0.8, va='center')

    txt_equal1 = ax1.text(5, 50, equal, fontsize=font_size, va='center')

    element_ex2 = f"${complex_exp_latex(e_0x, phi_x_pi)}$"
    element_ey2 = f"${complex_exp_latex(e_0y, phi_y_pi)}$"

    txt_element_ex2 = ax1.text(22, 45, element_ex2, fontsize=font_size, va='center')
    txt_element_ey2 = ax1.text(22, 55, element_ey2, fontsize=font_size, va='center')

    norm = ""
    txt_norm = ax1.text(10, 50, norm, fontsize=font_size, va='center')

    draw_bracket(ax1, 20, 20, 30, 16, "black")
    draw_bracket(ax1, 60, 20, 30, 16, "black")
    draw_bracket(ax1, 20, 50, 30, 16, "black")

    width_init, height_init = canvas.get_width_height()

    x = np.arange(x_min, x_max, 0.005)
    y0 = x * 0. + y_max
    z0 = x * 0. + z_min

    wave_e_0x = e_0x * np.cos((k * x - omega * t + phi_x_pi) * np.pi)
    plt_wave_e_0x, = ax0.plot(x, wave_e_0x, z0, color="blue", ls="--", linewidth=1)

    wave_e_0y = e_0y * np.cos((k * x - omega * t + phi_y_pi) * np.pi)
    plt_wave_e_0y, = ax0.plot(x, y0, wave_e_0y, color="green", ls="--", linewidth=1)

    plt_wave_e, = ax0.plot(x, wave_e_0x, wave_e_0y, color="red", ls="-", linewidth=2)

    for i_ in range(num_arrows):
        x_ = x_min + i_ * (x_max - x_min) / num_arrows
        y_ = scale_norm * e_0x * np.cos((k * x_ - omega * t + phi_x_pi) * np.pi)
        z_ = scale_norm * e_0y * np.cos((k * x_ - omega * t + phi_y_pi) * np.pi)
        arrow = Arrow3d(ax0, x_, 0., 0., 0., y_, z_, "red", 0.5, "-", 0.2)
        arrows.append(arrow)

    # ax0.legend(loc='lower right', fontsize=8)

    anim = animation.FuncAnimation(fig, update, interval=100, save_count=100)
    root.mainloop()
