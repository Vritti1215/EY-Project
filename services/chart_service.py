"""
Chart Service Module
=====================
Generates sales/financial charts as PNG images using Matplotlib + Seaborn.
Runs headless (Agg backend) so it works on servers with no display — this
is required on platforms like Render.

Design goals:
- Deterministic, server-side rendering. The AI model never touches image
  generation — it only ever gets to *describe* numbers that were already
  computed in Python (see insight_service.py). This avoids hallucinated
  totals in charts.
- Highly configurable: chart type, seaborn style/palette/context, figure
  size, and DPI can all be overridden per call or via Flask app config /
  environment variables.
- No dependency on persistent disk. Charts are returned as PNG bytes or a
  base64 data URI so they survive Render's free-tier ephemeral filesystem
  and can be embedded directly in a chat response and downloaded
  client-side with zero extra routes or storage.
"""

import io
import base64
import logging

import matplotlib
matplotlib.use('Agg')  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

CHART_TYPES = ('bar', 'barh', 'line', 'area', 'pie')

# ---------------------------------------------------------------------------
# Brand / theme palettes
# ---------------------------------------------------------------------------
# EY's brand palette (black, yellow, grey, white) is the primary theme for
# this app's charts. Named "additional colors" (blue, red, green) are also
# offered as single-hue alternatives for when a plain colored chart is
# wanted instead of the brand theme — these map to matplotlib's built-in
# sequential colormaps rather than a flat single color, so bar/line charts
# still get a readable light-to-dark gradient across categories.
#
# BRAND_PALETTES values are explicit hex lists. seaborn's sns.color_palette()
# / sns.barplot(palette=...) both cycle a short list of colors across
# however many categories are being plotted, so a 4-color theme works fine
# even with more than 4 bars.
BRAND_PALETTES = {
    'ey': ['#2E2E38', '#FFE600', '#9DA0A6', '#FFFFFF'],
    'ey_theme': ['#2E2E38', '#FFE600', '#9DA0A6', '#FFFFFF'],
    'black_yellow_grey_white': ['#2E2E38', '#FFE600', '#9DA0A6', '#FFFFFF'],
}

# Solid hex versions of the "additional colors", used (in this exact order)
# to auto-extend the brand palette once a chart has more categories than
# the brand palette has colors — e.g. a 7-slice pie automatically gets the
# 4 brand colors PLUS blue/red/green rather than repeating black/yellow/
# grey/white twice. Order matters: blue first, then red, then green.
EXTRA_BRAND_COLORS = ['#2563EB', '#DC2626', '#16A34A']  # blue, red, green

# Additional single-color options, layered on top of the brand theme.
# Mapped to matplotlib's built-in sequential colormaps (not a flat color)
# so multi-category charts still render with visible contrast.
COLOR_ALIASES = {
    'blue': 'Blues',
    'red': 'Reds',
    'green': 'Greens',
}

PALETTE_CHOICES = (
    'ey', 'blue', 'red', 'green',
    'deep', 'colorblind', 'viridis', 'mako', 'rocket', 'flare', 'crest',
    'coolwarm', 'pastel', 'bright', 'muted', 'dark',
)


def _brand_palette_for_count(base_colors, n_categories):
    """
    Auto-extend a brand palette to fit the number of bars/slices being
    drawn: use the brand colors as-is while there are enough of them, and
    once n_categories exceeds that, append the additional blue/red/green
    colors (in that order) before falling back to cycling from the start.
    This only applies to the brand/theme palette — an explicitly chosen
    single color (blue/red/green/etc.) is never auto-mixed with others.
    """
    if n_categories <= len(base_colors):
        return base_colors
    extended = list(base_colors) + EXTRA_BRAND_COLORS
    if n_categories <= len(extended):
        return extended
    # Still more categories than brand + extras: cycle the extended list.
    return [extended[i % len(extended)] for i in range(n_categories)]


class ChartService:
    """Configurable seaborn/matplotlib chart renderer."""

    def __init__(self, style='darkgrid', palette='ey', context='notebook',
                 dpi=150, figsize=(10, 6)):
        self.style = style
        self.palette = palette
        self.context = context
        self.dpi = dpi
        self.figsize = figsize

    @classmethod
    def from_app_config(cls, config):
        """Build a ChartService using Flask app config values, with fallbacks."""
        return cls(
            style=config.get('CHART_STYLE', 'darkgrid'),
            palette=config.get('CHART_PALETTE', 'ey'),
            context=config.get('CHART_CONTEXT', 'notebook'),
            dpi=config.get('CHART_DPI', 150),
            figsize=tuple(config.get('CHART_FIGSIZE', (10, 6))),
        )

    @staticmethod
    def _contrasting_text_color(background_hex):
        """
        Pick black or white text/gridline color based on the relative
        luminance of an explicit background color, so an instruction like
        "black background" doesn't also make the title/axis text
        unreadable black-on-black by leaving it at its old default.
        """
        try:
            hex_str = str(background_hex).lstrip('#')
            if len(hex_str) == 3:
                hex_str = ''.join(c * 2 for c in hex_str)
            r, g, b = (int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return '#111111' if luminance > 0.6 else '#FFFFFF'
        except Exception:
            return '#FFFFFF'

    @staticmethod
    def resolve_palette(name, n_categories=None):
        """
        Resolve a palette name into whatever seaborn/matplotlib actually
        expects: a hex color list for brand/theme palettes, a colormap
        name for the 'blue'/'red'/'green' aliases, or the name unchanged
        for any regular seaborn/matplotlib palette (viridis, mako, etc.).

        seaborn accepts either form (a list of colors or a colormap name
        string) anywhere a `palette=` argument is used, so callers don't
        need to know which kind they got back.

        When `name` resolves to a brand/theme palette and n_categories is
        given, the brand colors are auto-extended with the additional
        blue/red/green colors once there are more bars/slices than brand
        colors (see _brand_palette_for_count) — an explicit single-color
        choice (blue/red/green/etc.) is returned as-is and never mixed in.
        """
        if not name:
            return None
        key = str(name).strip().lower()
        if key in BRAND_PALETTES:
            base = BRAND_PALETTES[key]
            if n_categories:
                return _brand_palette_for_count(base, n_categories)
            return base
        if key in COLOR_ALIASES:
            return COLOR_ALIASES[key]
        return name

    def render_png_bytes(self, labels, values, chart_type='bar', title='Chart',
                          xlabel='', ylabel='', palette=None, style=None,
                          dpi=None, figsize=None, value_format='{:,.0f}', show_data_labels=True, show_variance=False, show_trend=False,
                          noise_variance=None, noise_seed=None,
                          background_color=None, series_color=None):
        """
        Render a chart to raw PNG bytes.

        Args:
            labels: category labels for the x-axis (or pie slice labels).
            values: numeric values, same length as labels.
            chart_type: one of CHART_TYPES ('bar', 'barh', 'line', 'area', 'pie').
            title, xlabel, ylabel: chart text.
            palette / style / dpi / figsize: per-call overrides of the
                instance defaults — this is the "max configurability" knob;
                any of these can be passed straight through from a request.
            value_format: format string applied to bar/line value labels.
            show_data_labels: show values on bars/lines when supported.
            show_variance: show a real-data +/- 1 standard deviation band for line/area.
            show_trend: show a linear trend line for line/area charts.
            noise_variance: optional fractional variance (e.g. 0.05 for +/-5%)
                to inject into the plotted values as a spec/tolerance-style
                perturbation — distinct from show_variance, which only draws
                a band over the real, unmodified data. When set, each value
                is jittered by Gaussian noise with std = |value| * noise_variance,
                and 1-sigma error bars are drawn (bar/line/barh) so the
                injected tolerance is visible, matching how a "+/-N% spec
                variance" requirement is usually represented in reports.
            noise_seed: optional RNG seed so the injected noise is
                reproducible across re-renders of the same request.
            background_color: explicit hex/name to set the figure and
                axes background to (e.g. "#000000" for "keep the
                background black"). When set, title/axis/tick text and
                the grid automatically switch to a light color so they
                stay readable against a dark background, and vice versa.
            series_color: explicit hex/name for the line/bars/area/pie
                wedges themselves (e.g. "#FFE600" for "the line in
                yellow"). Overrides `palette` for that one element when
                given — this is a specific single-color instruction, not
                a themed multi-color palette pick.

        Returns:
            bytes: PNG image data.
        """
        if chart_type not in CHART_TYPES:
            raise ValueError(f"Unsupported chart_type '{chart_type}'. Choose from {CHART_TYPES}.")
        if not labels or not values or len(labels) != len(values):
            raise ValueError("labels and values must be non-empty and of equal length.")

        noise_err = None
        if noise_variance:
            import numpy as np
            rng = np.random.default_rng(noise_seed)
            values = list(values)
            noise_err = [abs(v) * float(noise_variance) for v in values]
            values = [v + rng.normal(0, e) if e > 0 else v for v, e in zip(values, noise_err)]

        sns.set_style(style or self.style)
        sns.set_context(self.context)
        active_palette = self.resolve_palette(palette or self.palette, n_categories=len(labels))
        if series_color:
            # An explicit single-color instruction ("line in yellow",
            # "bars in red") overrides whatever palette/theme was
            # resolved above — every downstream branch (bar/line/area/
            # pie) reads `colors`/`bar_palette` derived from
            # active_palette, so overriding it here once is enough to
            # apply consistently without touching each chart-type branch.
            active_palette = [series_color]
        # Materialize to the exact number of categories needed so a short
        # brand/theme color list (e.g. the 4-color EY palette) cycles
        # cleanly across any number of bars without seaborn's "palette
        # has fewer/more values than needed" warning.
        bar_palette = sns.color_palette(active_palette, max(len(labels), 1))

        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=dpi or self.dpi)

        if background_color:
            # Auto-contrast title/axis/tick/grid text against whatever
            # background was requested, so "black background" doesn't
            # also silently make the labels unreadable black-on-black.
            fig.patch.set_facecolor(background_color)
            ax.set_facecolor(background_color)
            text_color = self._contrasting_text_color(background_color)
            ax.title.set_color(text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)
            ax.grid(True, color=text_color, alpha=0.15)

        try:
            if chart_type == 'bar':
                sns.barplot(x=list(labels), y=list(values), hue=list(labels),
                            palette=bar_palette, legend=False, ax=ax)
                if noise_err:
                    ax.errorbar(x=list(range(len(labels))), y=list(values), yerr=noise_err,
                                fmt='none', ecolor='#2E2E38', elinewidth=1.2, capsize=4, alpha=0.8,
                                label=f'±{noise_variance*100:.0f}% variance')
                    ax.legend(fontsize=8)
                if show_data_labels:
                    for i, v in enumerate(values):
                        ax.text(i, v, value_format.format(v), ha='center', va='bottom', fontsize=9)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

            elif chart_type == 'barh':
                sns.barplot(y=list(labels), x=list(values), hue=list(labels),
                            palette=bar_palette, legend=False, ax=ax)
                if noise_err:
                    ax.errorbar(y=list(range(len(labels))), x=list(values), xerr=noise_err,
                                fmt='none', ecolor='#2E2E38', elinewidth=1.2, capsize=4, alpha=0.8,
                                label=f'±{noise_variance*100:.0f}% variance')
                    ax.legend(fontsize=8)
                ax.set_xlabel(ylabel)
                ax.set_ylabel(xlabel)

            elif chart_type == 'line':
                colors = sns.color_palette(active_palette, 1)
                sns.lineplot(x=list(labels), y=list(values), marker='o',
                             color=colors[0], ax=ax)
                if noise_err:
                    ax.errorbar(x=list(labels), y=list(values), yerr=noise_err,
                                fmt='none', ecolor=colors[0], elinewidth=1.2, capsize=4, alpha=0.7,
                                label=f'±{noise_variance*100:.0f}% variance')
                    ax.legend(fontsize=8)
                if show_data_labels:
                    for i, v in enumerate(values):
                        ax.text(i, v, value_format.format(v), ha='center', va='bottom', fontsize=9)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                if show_variance and len(values) > 1:
                    import numpy as np
                    mean = float(np.mean(values))
                    std = float(np.std(values))
                    ax.axhline(mean, linestyle='--', linewidth=1, alpha=0.55, label='Mean')
                    ax.axhspan(mean - std, mean + std, alpha=0.12, label='±1 SD')
                    ax.legend(fontsize=8)
                if show_trend and len(values) > 1:
                    import numpy as np
                    x = np.arange(len(values), dtype=float)
                    slope, intercept = np.polyfit(x, np.asarray(values, dtype=float), 1)
                    ax.plot(x, slope * x + intercept, linestyle='--', linewidth=1.2, label='Trend')
                    ax.legend(fontsize=8)

            elif chart_type == 'area':
                colors = sns.color_palette(active_palette, 1)
                x_positions = range(len(labels))
                ax.fill_between(x_positions, values, alpha=0.4, color=colors[0])
                ax.plot(x_positions, values, marker='o', color=colors[0])
                ax.set_xticks(list(x_positions))
                ax.set_xticklabels(labels)
                if show_data_labels:
                    for i, v in enumerate(values):
                        ax.text(i, v, value_format.format(v), ha='center', va='bottom', fontsize=9)
                if show_variance and len(values) > 1:
                    import numpy as np
                    mean = float(np.mean(values))
                    std = float(np.std(values))
                    ax.axhspan(mean - std, mean + std, alpha=0.12, label='±1 SD')
                    ax.axhline(mean, linestyle='--', linewidth=1, alpha=0.55, label='Mean')
                    ax.legend(fontsize=8)
                if show_trend and len(values) > 1:
                    import numpy as np
                    x = np.arange(len(values), dtype=float)
                    slope, intercept = np.polyfit(x, np.asarray(values, dtype=float), 1)
                    ax.plot(x, slope * x + intercept, linestyle='--', linewidth=1.2, label='Trend')
                    ax.legend(fontsize=8)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

            elif chart_type == 'pie':
                colors = sns.color_palette(active_palette, len(labels))
                ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors,
                       startangle=90)
                ax.axis('equal')

            ax.set_title(title, fontsize=13, fontweight='bold')
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi or self.dpi, facecolor=fig.get_facecolor())
            buf.seek(0)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Chart rendering failed: {e}")
            raise
        finally:
            plt.close(fig)

    def render_base64(self, *args, **kwargs):
        """Same as render_png_bytes but returns a base64 PNG data URI string,
        ready to drop straight into an <img src="..."> or a download link."""
        png_bytes = self.render_png_bytes(*args, **kwargs)
        encoded = base64.b64encode(png_bytes).decode('ascii')
        return f'data:image/png;base64,{encoded}'