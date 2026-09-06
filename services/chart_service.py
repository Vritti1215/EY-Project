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
CUSTOM_PALETTES = {
    # EY-style base palette: black / yellow / grey / white.
    'ey': ['#2E2E38', '#FFE600', '#6B6B76', '#D9D9DE', '#FFFFFF'],
    'yellow': ['#FFE600', '#E6CF00', '#2E2E38', '#6B6B76'],
    'black': ['#2E2E38', '#4A4A52', '#6B6B76', '#8A8A92'],
    'grey': ['#4A4A52', '#6B6B76', '#9A9AA2', '#D9D9DE'],
    'white': ['#FFFFFF', '#D9D9DE', '#9A9AA2', '#4A4A52'],
    'blue': ['#2563EB', '#60A5FA', '#1D4ED8', '#93C5FD'],
    'red': ['#DC2626', '#F87171', '#B91C1C', '#FCA5A5'],
    'green': ['#16A34A', '#4ADE80', '#15803D', '#86EFAC'],
    'multi': ['#FFE600', '#2563EB', '#DC2626', '#16A34A', '#2E2E38', '#6B6B76'],
}

def _palette_values(name, n):
    key = str(name or '').lower()
    if key in CUSTOM_PALETTES:
        base = CUSTOM_PALETTES[key]
        return [base[i % len(base)] for i in range(max(1, n))]
    return sns.color_palette(name or 'viridis', max(1, n))


class ChartService:
    """Configurable seaborn/matplotlib chart renderer."""

    def __init__(self, style='darkgrid', palette='viridis', context='notebook',
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
            palette=config.get('CHART_PALETTE', 'viridis'),
            context=config.get('CHART_CONTEXT', 'notebook'),
            dpi=config.get('CHART_DPI', 150),
            figsize=tuple(config.get('CHART_FIGSIZE', (10, 6))),
        )

    def render_png_bytes(self, labels, values, chart_type='bar', title='Chart',
                          xlabel='', ylabel='', palette=None, style=None,
                          dpi=None, figsize=None, value_format='{:,.0f}', show_data_labels=True, show_variance=False, show_trend=False):
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

        Returns:
            bytes: PNG image data.
        """
        if chart_type not in CHART_TYPES:
            raise ValueError(f"Unsupported chart_type '{chart_type}'. Choose from {CHART_TYPES}.")
        if not labels or not values or len(labels) != len(values):
            raise ValueError("labels and values must be non-empty and of equal length.")

        sns.set_style(style or self.style)
        sns.set_context(self.context)
        active_palette = palette or self.palette
        palette_values = _palette_values(active_palette, len(labels))

        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=dpi or self.dpi)

        try:
            if chart_type == 'bar':
                sns.barplot(x=list(labels), y=list(values), hue=list(labels),
                            palette=palette_values, legend=False, ax=ax)
                if show_data_labels:
                    for i, v in enumerate(values):
                        ax.text(i, v, value_format.format(v), ha='center', va='bottom', fontsize=9)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

            elif chart_type == 'barh':
                sns.barplot(y=list(labels), x=list(values), hue=list(labels),
                            palette=palette_values, legend=False, ax=ax)
                ax.set_xlabel(ylabel)
                ax.set_ylabel(xlabel)

            elif chart_type == 'line':
                colors = _palette_values(active_palette, 1)
                sns.lineplot(x=list(labels), y=list(values), marker='o',
                             color=colors[0], ax=ax)
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
                colors = _palette_values(active_palette, 1)
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
                colors = _palette_values(active_palette, len(labels))
                ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors,
                       startangle=90)
                ax.axis('equal')

            ax.set_title(title, fontsize=13, fontweight='bold')
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi or self.dpi)
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