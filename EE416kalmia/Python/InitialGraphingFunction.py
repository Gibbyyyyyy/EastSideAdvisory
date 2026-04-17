# Use this to run it:
# from signal_viewer import show_samples
# show_samples(samples)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.ticker import MultipleLocator   # For minor grid axis
from matplotlib.widgets import Button, CheckButtons, TextBox    # Checkboxes

# Load all of the toolbars tools except subplots (unused)
if getattr(NavigationToolbar2, "toolitems", None):
    NavigationToolbar2.toolitems = tuple(
        item for item in NavigationToolbar2.toolitems if item[0] != "Subplots"
    )

# Dark mode color scheme
_DARK = {
    "fig_bg":        "#1a202c",
    "plot_bg":       "#11151d",
    "grid":          "#252d3d",
    "spine":         "#3a4556",
    "xaxis_line":    "#4a5568",
    "tick_minor":    "#2d3748",
    "text_primary":  "#e2e8f0",
    "text_secondary":"#a0aec0",
    "control_bg":    "#4a5568",
    "control_hover": "#5a677d",
    "signal_left":   "#ff6b6b",
    "signal_right":  "#339af0",
    "marker_kal":    "#20c997",
    "marker_met":    "#cc5de8",
    "marker_win":    "#94a3b8",
    "legend_text":   "#cbd5e1",
    "no_signal_text":"#475569",
}

# Light mode color scheme
_LIGHT = {
    "fig_bg":        "#f1f5f9",
    "plot_bg":       "#ffffff",
    "grid":          "#e2e8f0",
    "spine":         "#94a3b8",
    "xaxis_line":    "#64748b",
    "tick_minor":    "#cbd5e1",
    "text_primary":  "#0f172a",
    "text_secondary":"#64748b",
    "control_bg":    "#e2e8f0",
    "control_hover": "#cbd5e1",
    "signal_left":   "#d93832",
    "signal_right":  "#1d78c1",
    "marker_kal":    "#0d9488",
    "marker_met":    "#a632bd",
    "marker_win":    "#64748b",
    "legend_text":   "#334155",
    "no_signal_text":"#94a3b8",
}

# Key bindings: key -> action name
_KEY_BINDINGS = {
    "left": "prev",
    "a": "prev",
    "right": "next",
    "d": "next",
    "home": "first",
    "l": "toggle_left",
    "r": "toggle_right",
    " ": "run",
}


# Turns input into a plain list of sample numbers.
def _normalize_samples(samples):
    return [samples[k] for k in sorted(samples)]


# This passes into the graphing system, making it simpler for things like marking the WF
def _as_1d_array(value):
    if value is None:  # Guard for if a zero value is passed in
        return None
    return np.asarray(value).reshape(-1)


# Finds the filepath of whatever signal is being observed. Allows it to be labeled in the graph eg: wave001.csv
def _display_path(sample):
    return sample.path.name


# Turns a number into a clean/readable string eg 12.300 -> 12.3
def _format_value(value, suffix=""):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.1f}" + suffix  # CHANGE THIS LINE TO ADJUST SIG FIGS IN TABLES /////////////////////////////////////////////////////////////////
    return f"{value}{suffix}"


# This function calculates the overall percentage of samples used
def _compute_side_usage_percent(samples, attr):
    used = sum(1 for sample in samples if getattr(sample, attr) == 1)
    return 100 * used / len(samples)


class SignalViewer:
    # Sets up the viewer window, all UI elements, and loads the first sample
    def __init__(
        self,
        samples,
        title="MetKal WF Detection",
        sample_rate_hz=3.5e6,
        run_interval_ms=500,  # Pause time for each signal when doing "run sheet." If changing, see show_samples at the bottom
    ):
        self.samples = _normalize_samples(samples)
        if not self.samples:
            raise ValueError("SignalViewer received no samples.")

        self.sample_rate_hz = float(sample_rate_hz)
        self.run_interval_ms = int(run_interval_ms)
        self.index = 0
        self.show_left = True
        self.show_right = True
        self.show_kal = True
        self.show_met = True
        self.show_win_left = True
        self.show_win_right = True
        self.is_running = False
        self.is_light_mode = False

        # Pre-compute sheet-level statistics
        self._kalmia_usage_pct_L = _compute_side_usage_percent(self.samples, "statusL")
        self._kalmia_usage_pct_R = _compute_side_usage_percent(self.samples, "statusR")
        self._metriguard_usage_pct_L = _compute_side_usage_percent(self.samples, "metStatusL")
        self._metriguard_usage_pct_R = _compute_side_usage_percent(self.samples, "metStatusR")

        self.fig = plt.figure(figsize=(16.8, 8.8))
        self.fig.patch.set_facecolor(_DARK["fig_bg"])
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        # Create main layout of the viewer (header, main graph, info area)
        self.ax_header = self.fig.add_axes([0.047, 0.92, 0.943, 0.05])
        self.ax_header.axis("off")
        self.ax = self.fig.add_axes([0.047, 0.30, 0.923, 0.59])  # The graph
        self.ax_table1 = self.fig.add_axes([0.40, 0.02, 0.2, 0.16])  # Sample table
        self.ax_table1.axis("off")
        self.ax_table2 = self.fig.add_axes([0.125, 0.02, 0.10, 0.16])  # Percent table
        self.ax_table2.axis("off")

        # Set sizes/spacing for start/stop/run/sample controls
        btn_w = 0.065  # button width
        btn_right = 0.875  # right edge
        btn_left = btn_right - btn_w  # left edge
        sample_label_w = 0.042  # "Sample" text label width
        gap = 0.008  # vertical gap between rows

        # Majorly redundant but allows for quick changes
        top_anchor = 0.180 + 0.045  # top of Kal row = 0.225
        bot_anchor = 0.045  # bottom of arrow row
        total_h = top_anchor - bot_anchor
        btn_h = (total_h - 2 * gap) / 3  # height of each row

        row1_y = top_anchor - btn_h  # Run Sheet
        row2_y = row1_y - gap - btn_h  # First
        row3_y = bot_anchor  # Sample

        self.ax_run = self.fig.add_axes([btn_left, row1_y, btn_w, btn_h])
        self.ax_first = self.fig.add_axes([btn_left, row2_y, btn_w, btn_h])
        self.ax_sample_label = self.fig.add_axes([btn_left, row3_y, sample_label_w, btn_h])
        self.ax_jump = self.fig.add_axes([btn_left + sample_label_w, row3_y, btn_w - sample_label_w, btn_h])
        self.ax_sample_label.axis("off")

        # Checkboxes and arrow buttons location mapping
        self.ax_check_kal = self.fig.add_axes([0.880, 0.180, 0.045, 0.045])
        self.ax_check_met = self.fig.add_axes([0.925, 0.180, 0.045, 0.045])
        self.ax_check_left = self.fig.add_axes([0.880, 0.135, 0.045, 0.045])
        self.ax_check_right = self.fig.add_axes([0.925, 0.135, 0.045, 0.045])
        self.ax_check_winL = self.fig.add_axes([0.880, 0.090, 0.045, 0.045])
        self.ax_check_winR = self.fig.add_axes([0.925, 0.090, 0.045, 0.045])
        self.ax_prev = self.fig.add_axes([0.880, 0.045, 0.045, 0.045])
        self.ax_next = self.fig.add_axes([0.925, 0.045, 0.045, 0.045])
        self.ax_check_light = self.fig.add_axes([0.880, 0.955, 0.090, 0.045])

        # Make the buttons clickable/usable
        self.btn_run = Button(self.ax_run, "Run Sheet")
        self.btn_first = Button(self.ax_first, "First")
        self.check_left = CheckButtons(self.ax_check_left, ["LHS"], [True])
        self.check_right = CheckButtons(self.ax_check_right, ["RHS"], [True])
        self.check_kal = CheckButtons(self.ax_check_kal, ["Kal"], [True])
        self.check_met = CheckButtons(self.ax_check_met, ["Met"], [True])
        self.check_win_left = CheckButtons(self.ax_check_winL, ["WL"], [True])
        self.check_win_right = CheckButtons(self.ax_check_winR, ["WR"], [True])
        self.check_light = CheckButtons(self.ax_check_light, ["Light Mode"], [False])
        self.btn_prev = Button(self.ax_prev, "←")
        self.btn_next = Button(self.ax_next, "→")
        self.txt_jump = TextBox(self.ax_jump, "", initial=self._sample_label(self.samples[0]))

        # Makes checkboxes/fonts look nice
        self._style_controls()

        # Connect buttons to their intended use
        self.btn_run.on_clicked(self._toggle_run_sheet)
        self.btn_first.on_clicked(self._first_sample)
        self.btn_prev.on_clicked(self._prev_sample)
        self.btn_next.on_clicked(self._next_sample)
        self.check_left.on_clicked(self._on_left_check_clicked)
        self.check_right.on_clicked(self._on_right_check_clicked)
        self.check_kal.on_clicked(self._on_kal_check_clicked)
        self.check_met.on_clicked(self._on_met_check_clicked)
        self.check_win_left.on_clicked(self._on_win_left_check_clicked)
        self.check_win_right.on_clicked(self._on_win_right_check_clicked)
        self.check_light.on_clicked(self._on_light_mode_clicked)
        self.txt_jump.on_submit(self._jump_to_sample)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

        # Create timer for run sheet mode
        self.run_timer = self.fig.canvas.new_timer(interval=self.run_interval_ms)
        self.run_timer.add_callback(self._run_sheet_step)

        self._draw_current_sample()

    # Opens the matplotlib window and hands control over to it
    def show(self):
        plt.show()

    # Identify if the system is in light or dark mode
    def _theme(self):
        return _LIGHT if self.is_light_mode else _DARK

    # Scales a base font size proportionally to the current figure size
    def _scaled_font(self, base_size):
        w, h = self.fig.get_size_inches()
        scale = min(w / 16.8, h / 8.8)  # 16.8x8.8 is the designed reference size
        return max(6, round(base_size * scale))

    # Font and label details
    def _style_controls(self):
        t = self._theme()
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "axes.labelsize": self._scaled_font(14),
                "xtick.labelsize": self._scaled_font(11),
                "ytick.labelsize": self._scaled_font(11),
                "legend.fontsize": self._scaled_font(10),
            }
        )

        self.fig.patch.set_facecolor(t["fig_bg"])

        for ax in (
            self.ax_run, self.ax_first, self.ax_jump,
            self.ax_check_left, self.ax_check_right,
            self.ax_check_kal, self.ax_check_met,
            self.ax_check_winL, self.ax_check_winR,
            self.ax_check_light, self.ax_prev, self.ax_next,
        ):
            ax.set_facecolor(t["control_bg"])

        for btn in (self.btn_run, self.btn_first, self.btn_prev, self.btn_next):
            btn.color = t["control_bg"]
            btn.hovercolor = t["control_hover"]
            btn.label.set_fontsize(self._scaled_font(12))
            btn.label.set_fontweight("bold")
            btn.label.set_fontstyle("italic")
            btn.label.set_color(t["text_primary"])

        # Arrow buttons get bigger font + no italic
        self.btn_prev.label.set_fontsize(self._scaled_font(18))
        self.btn_next.label.set_fontsize(self._scaled_font(18))
        self.btn_prev.label.set_fontstyle("normal")
        self.btn_next.label.set_fontstyle("normal")

        self.ax_sample_label.clear()
        self.ax_sample_label.axis("off")
        self.ax_sample_label.set_facecolor(t["fig_bg"])
        self.ax_sample_label.text(
            0.0, 0.5, "Sample",
            ha="left", va="center",
            fontsize=self._scaled_font(12), fontweight="bold", fontstyle="italic", color=t["text_primary"],
        )

        for widget in (self.check_left, self.check_right, self.check_kal,
                       self.check_met, self.check_win_left, self.check_win_right,
                       self.check_light):
            for label in getattr(widget, "labels", []):
                label.set_fontsize(self._scaled_font(12))
                label.set_fontweight("bold")
                label.set_fontstyle("italic")
                label.set_color(t["text_primary"])
            for line_pair in getattr(widget, "lines", []):
                for line in line_pair:
                    line.set_linewidth(2.8)
                    line.set_color(t["text_primary"])

    # Returns whichever sample object is currently being displayed
    def _current_sample(self):
        return self.samples[self.index]

    # Returns the sample number as a string for display purposes
    def _sample_label(self, sample):
        return str(sample.sample_number)

    # Updates the sample number text box without triggering the jump logic
    def _set_jump_box_value(self, value):
        try:
            previous = self.txt_jump.eventson
            self.txt_jump.eventson = False
            self.txt_jump.set_val(value)
            self.txt_jump.eventson = previous
        except Exception:
            pass

    # Create an x axis of time values since initially it only passes in as a list of sample numbers
    def _x_values_us(self, sample, desired_len):
        return sample.n[:desired_len] / self.sample_rate_hz * 1e6

    # Convert a sample value to microseconds
    def _idx_to_us(self, idx):
        if idx is None:
            return None
        try:
            return float(idx) / self.sample_rate_hz * 1e6
        except Exception:
            return None

    # Calculates the delta UPT between each side for a given sample
    def _delta_us(self, a, b):
        t_a = self._idx_to_us(a)
        t_b = self._idx_to_us(b)
        if t_a is None or t_b is None:  # Prevents zero values from breaking viewer
            return None
        return t_a - t_b

    # Draw a vertical marker line at the time position corresponding to sample index `idx`.
    def _draw_marker(self, idx, label, color, linestyle='-', linewidth=2):
        if not idx:
            return
        x_us = self._idx_to_us(idx)
        self.ax.axvline(x_us, linestyle=linestyle, linewidth=linewidth, color=color, alpha=0.95, label=label, zorder=2)

    # Plots one side's signal and all its markers. side = "L" or "R".
    def _plot_side(self, side, y, x_full, t, show_win):
        if y is None:
            return False

        sample = self._current_sample()
        is_left = side == "L"
        signal_color = t["signal_left"] if is_left else t["signal_right"]
        signal_label = "Left Signal" if is_left else "Right Signal"

        wf = sample.wavefrontL if is_left else sample.wavefrontR
        met = sample.metSelectionL if is_left else sample.metSelectionR
        status = sample.statusL if is_left else sample.statusR
        edge1 = sample.edge1L if is_left else sample.edge1R
        edge2 = sample.edge2L if is_left else sample.edge2R

        x = x_full[: len(y)]
        self.ax.plot(x, y, color=signal_color, linewidth=2.4, label=signal_label, zorder=3)
        if self.show_kal:
            self._draw_marker(wf, f"Kalmia {side}", t["marker_kal"], linewidth=2.8)
        if self.show_met:
            self._draw_marker(met, f"Metriguard {side}", t["marker_met"], linewidth=2.8)
        if show_win and status == 1:
            self._draw_marker(edge1, f"Window Edges {side}", t["marker_win"], linestyle="--", linewidth=1.7)
            self._draw_marker(edge2, f"Window Edges {side}", t["marker_win"], linestyle="--", linewidth=1.7)
        return True

    # Draws both info tables into their axes
    def _draw_tables(self, sample, t):
        wave_l_us = self._idx_to_us(sample.wavefrontL)
        wave_r_us = self._idx_to_us(sample.wavefrontR)
        met_l_us = self._idx_to_us(sample.metSelectionL)
        met_r_us = self._idx_to_us(sample.metSelectionR)
        delta_l = self._delta_us(sample.wavefrontL, sample.metSelectionL)
        delta_r = self._delta_us(sample.wavefrontR, sample.metSelectionR)

        # Table 1 - per-sample values: rows=metrics, cols=Left/Right
        t1_data = [
            [_format_value(wave_l_us, " µs"), _format_value(wave_r_us, " µs")],
            [_format_value(met_l_us, " µs"), _format_value(met_r_us, " µs")],
            [_format_value(delta_l, " µs"), _format_value(delta_r, " µs")],
            [_format_value(sample.snrL), _format_value(sample.snrR)],
        ]
        t1_rows = ["Kalmia", "Metriguard", "Delta UPT", "SNR"]
        t1_cols = ["Left", "Right"]

        tbl1 = self.ax_table1.table(
            cellText=t1_data,
            rowLabels=t1_rows,
            colLabels=t1_cols,
            loc="center",
            cellLoc="center",
        )
        tbl1.auto_set_font_size(False)
        tbl1.set_fontsize(self._scaled_font(10))
        tbl1.scale(1, 1.495)  # 1.3 * 1.15

        # Table 2 - sheet-level stats: rows=Kalmia/Metriguard, col=Samples Used
        t2_data = [
            [_format_value(self._kalmia_usage_pct_L, "%"), _format_value(self._kalmia_usage_pct_R, "%")],
            [_format_value(self._metriguard_usage_pct_L, "%"), _format_value(self._metriguard_usage_pct_R, "%")],
        ]
        t2_rows = ["Kalmia", "Metriguard"]
        t2_cols = ["LHS Used", "RHS Used"]

        tbl2 = self.ax_table2.table(
            cellText=t2_data,
            rowLabels=t2_rows,
            colLabels=t2_cols,
            loc="center",
            cellLoc="center",
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(self._scaled_font(10))
        tbl2.scale(1, 1.495)  # 1.3 * 1.15

        # Apply theme colors to both tables
        for tbl in (tbl1, tbl2):
            for (row, col), cell in tbl.get_celld().items():
                cell.set_facecolor(t["fig_bg"])
                cell.set_edgecolor(t["spine"])
                cell.set_linewidth(1.15)
                cell.set_text_props(
                    color=t["text_primary"],
                    va="center",
                    fontweight="bold",
                    fontstyle="italic",
                )

    # Clears the plot and redraws everything for the currently selected sample
    def _draw_current_sample(self):
        sample = self._current_sample()
        t = self._theme()
        self.ax.clear()
        self.ax_table1.clear()
        self.ax_table1.axis("off")
        self.ax_table2.clear()
        self.ax_table2.axis("off")
        self.ax_header.clear()
        self.ax_header.axis("off")

        y_left = _as_1d_array(sample.amplitudeL)  # Load signal data
        y_right = _as_1d_array(sample.amplitudeR)
        max_len = max(len(y_left) if y_left is not None else 0, len(y_right) if y_right is not None else 0)
        x_us_full = self._x_values_us(sample, max_len) if max_len else np.array([])

        plotted_left = self._plot_side("L", y_left, x_us_full, t, self.show_win_left) if self.show_left else False
        plotted_right = self._plot_side("R", y_right, x_us_full, t, self.show_win_right) if self.show_right else False
        plotted = plotted_left or plotted_right

        self.ax.set_facecolor(t["plot_bg"])
        self.ax.axhline(0, color=t["xaxis_line"], linewidth=1.0, alpha=0.75, zorder=1)
        self.ax.grid(True, color=t["grid"], linewidth=0.7, alpha=0.55)
        for spine in self.ax.spines.values():
            spine.set_color(t["spine"])

        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlabel("Time (µs)")
        self.ax.yaxis.label.set_color(t["text_primary"])
        self.ax.xaxis.label.set_color(t["text_primary"])
        self.ax.tick_params(colors=t["text_primary"])

        if len(x_us_full):
            self.ax.set_xlim(float(x_us_full[0]), float(x_us_full[-1]))

        # Minor legend ticks along x axis
        self.ax.xaxis.set_minor_locator(MultipleLocator(20))
        self.ax.tick_params(axis="x", which="minor", length=4, width=0.8, color=t["tick_minor"])

        # Symmetric y-axis, fixed at 0, rounded up to nearest 0.2, sets minimum zoom
        all_y = []
        if self.show_left and y_left is not None:
            all_y.extend(y_left)
        if self.show_right and y_right is not None:
            all_y.extend(y_right)
        if all_y:
            raw_max = np.max(np.abs(all_y))
            rounded_max = np.ceil(raw_max / 0.2) * 0.2
        else:
            rounded_max = 0.0
        rounded_max = max(rounded_max, 2.0)  # never smaller than this value /////////////////////////////////////////////////////
        self.ax.set_ylim(-rounded_max, rounded_max)

        if plotted:  # Build legend
            handles, labels = self.ax.get_legend_handles_labels()
            seen = set()
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label not in seen:
                    seen.add(label)
                    unique_handles.append(handle)
                    unique_labels.append(label)
            self.ax.legend(
                unique_handles,
                unique_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.105),
                ncol=min(6, len(unique_labels)),
                frameon=False,
                labelcolor=t["legend_text"],
                handlelength=1.3,
                handletextpad=0.35,
                columnspacing=0.9,
                borderaxespad=0.0,
            )
        else:
            self.ax.text(0.5, 0.5, "No visible channel selected or no signal data available.",
                transform=self.ax.transAxes, ha="center", va="center", fontsize=self._scaled_font(13), color=t["no_signal_text"])

        header = f"Ultrasonic Wavefront Detection - {_display_path(sample)}"
        self.ax_header.text(0.5, 0.5, header, ha="center", va="center", fontsize=self._scaled_font(18), fontweight="normal", fontstyle="italic", color=t["text_primary"])

        self._draw_tables(sample, t)

        self._set_jump_box_value(self._sample_label(sample))
        self.fig.canvas.draw_idle()

    # Goes to the previous sample
    def _prev_sample(self, _event=None):
        self._stop_run_sheet()
        self.index = (self.index - 1) % len(self.samples)
        self._draw_current_sample()

    # Goes to the next sample
    def _next_sample(self, _event=None):
        self._stop_run_sheet()
        self.index = (self.index + 1) % len(self.samples)
        self._draw_current_sample()

    # Jumps back to the first sample
    def _first_sample(self, _event=None):
        self._stop_run_sheet()
        self.index = 0
        self._draw_current_sample()

    # Reads the text box input and jumps to the matching sample number
    def _jump_to_sample(self, text):
        query = text.strip()
        if not query:
            self._set_jump_box_value(self._sample_label(self._current_sample()))
            return

        try:
            query = int(query)
        except ValueError:
            self._set_jump_box_value(self._sample_label(self._current_sample()))
            return

        for index, sample in enumerate(self.samples):
            if int(sample.sample_number) == query:
                self._stop_run_sheet()
                self.index = index
                self._draw_current_sample()
                return

        self._set_jump_box_value(self._sample_label(self._current_sample()))

    # Starts or stops the auto-run that steps through all samples
    def _toggle_run_sheet(self, _event=None):
        if self.is_running:
            self._stop_run_sheet()
            self.fig.canvas.draw_idle()
            return
        self.is_running = True
        self.btn_run.label.set_text("Stop")
        self.run_timer.start()
        self.fig.canvas.draw_idle()

    # Stops the auto-run and resets the button label
    def _stop_run_sheet(self):
        if self.is_running:
            self.is_running = False
            self.run_timer.stop()
            self.btn_run.label.set_text("Run Sheet")

    # Advances one sample forward each time the run timer hits
    def _run_sheet_step(self):
        if not self.is_running:
            return
        if self.index >= len(self.samples) - 1:
            self._stop_run_sheet()
            self.fig.canvas.draw_idle()
            return
        self.index += 1
        self._draw_current_sample()

    # Called when the L checkbox is toggled; updates the display
    def _on_left_check_clicked(self, _label=None):
        self.show_left = bool(self.check_left.get_status()[0])
        if not self.show_left and self.show_win_left:
            self.show_win_left = False
            self.check_win_left.set_active(0)
        self._draw_current_sample()

    # Called when the R checkbox is toggled; updates the display
    def _on_right_check_clicked(self, _label=None):
        self.show_right = bool(self.check_right.get_status()[0])
        if not self.show_right and self.show_win_right:
            self.show_win_right = False
            self.check_win_right.set_active(0)
        self._draw_current_sample()

    # Called when Kal checkbox is toggled; updates the display
    def _on_kal_check_clicked(self, _label=None):
        self.show_kal = bool(self.check_kal.get_status()[0])
        self._draw_current_sample()

    # Called when Met checkbox is toggled; updates the display
    def _on_met_check_clicked(self, _label=None):
        self.show_met = bool(self.check_met.get_status()[0])
        self._draw_current_sample()

    # Called when the window left checkbox is toggled; updates the display
    def _on_win_left_check_clicked(self, _label=None):
        self.show_win_left = bool(self.check_win_left.get_status()[0])
        self._draw_current_sample()

    # Called when the window right checkbox is toggled; updates the display
    def _on_win_right_check_clicked(self, _label=None):
        self.show_win_right = bool(self.check_win_right.get_status()[0])
        self._draw_current_sample()

    # Called when the light mode checkbox is toggled; rebuilds styles and redraws
    def _on_light_mode_clicked(self, _label=None):
        self.is_light_mode = bool(self.check_light.get_status()[0])
        self._style_controls()
        self._draw_current_sample()

    # Handles keyboard shortcuts for navigating samples
    def _on_key_press(self, event):
        action = _KEY_BINDINGS.get(event.key)
        if action == "prev":
            self._prev_sample()
        elif action == "next":
            self._next_sample()
        elif action == "first":
            self._first_sample()
        elif action == "toggle_left":
            self.check_left.set_active(0)
        elif action == "toggle_right":
            self.check_right.set_active(0)
        elif action == "run":
            self._toggle_run_sheet()

    # Fires when the window is resized — rescales fonts and redraws
    def _on_resize(self, _event=None):
        self._style_controls()
        self._draw_current_sample()


# Entry point - creates the viewer and opens the window
def show_samples(
    samples,
    title="Ultrasonic Signal Viewer",
    sample_rate_hz=3.5e6,
    run_interval_ms=200,  # CHANGE THE PACE OF THE RUN SHEET FUNCTION HERE //////////////////////////////////////////////////////////////////////////////////
):
    viewer = SignalViewer(
        samples=samples,
        title=title,
        sample_rate_hz=sample_rate_hz,
        run_interval_ms=run_interval_ms,
    )
    viewer.show()
    return viewer
