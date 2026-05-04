"""
Shared UI helpers: CSS injection, Plotly theme, and reusable HTML components.
Imported by every page to maintain a consistent dark fintech look.
"""
import streamlit as st

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
THEME = {
    "bg":           "#0A0A0F",
    "card":         "#111118",
    "border":       "#1E1E2E",
    "accent":       "#2563EB",
    "bullish":      "#00C896",
    "bearish":      "#FF4444",
    "text":         "#F1F5F9",
    "muted":        "#94A3B8",
    "warn":         "#F59E0B",
}

# Plotly modebar config: keep only zoom and pan
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToKeep": ["zoom2d", "pan2d", "resetScale2d"],
    "scrollZoom": True,
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Background ─────────────────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: #0A0A0F !important;
}
[data-testid="stHeader"] {
    background-color: #0A0A0F !important;
    border-bottom: 1px solid #1E1E2E !important;
}
[data-testid="stMain"] {
    background-color: #0A0A0F !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #111118 !important;
    border-right: 1px solid #1E1E2E !important;
}
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"] {
    background-color: #111118 !important;
}
[data-testid="stSidebarNavLink"] {
    color: #94A3B8 !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.75rem !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background-color: rgba(37, 99, 235, 0.12) !important;
    color: #2563EB !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: rgba(37, 99, 235, 0.18) !important;
    color: #2563EB !important;
    font-weight: 600 !important;
}

/* ── Typography ──────────────────────────────────────────────────────────── */
p, span, div, td, th, li {
    color: #F1F5F9;
}
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}
label,
.stTextInput label,
.stSelectbox label,
.stRadio label,
.stSlider label,
.stTextArea label,
.stFileUploader label {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    padding: 0.55rem 1.3rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    color: #ffffff !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 0 28px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #2563EB !important;
    border: 1.5px solid #2563EB !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(37, 99, 235, 0.1) !important;
    box-shadow: 0 0 18px rgba(37, 99, 235, 0.25) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background-color: #1E1E2E !important;
    color: #F1F5F9 !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.35) !important;
}
.stSelectbox > div > div {
    background-color: #1E1E2E !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
}
.stSelectbox [data-baseweb="select"] {
    background-color: #1E1E2E !important;
}
.stTextArea textarea {
    background-color: #1E1E2E !important;
    color: #F1F5F9 !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
}
.stRadio > div > div > div > label { color: #F1F5F9 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #2563EB !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: #111118 !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] > div {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] > div {
    color: #F1F5F9 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] > details > summary {
    background-color: #111118 !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stExpander"] > details[open] > summary {
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stExpander"] > details > div {
    background-color: #111118 !important;
    border: 1px solid #1E1E2E !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 1rem !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background-color: #111118 !important;
    border: 2px dashed #2563EB !important;
    border-radius: 10px !important;
    transition: background 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(37, 99, 235, 0.06) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; }
.stSuccess { border-left: 3px solid #00C896 !important; }
.stInfo    { border-left: 3px solid #2563EB !important; }
.stWarning { border-left: 3px solid #F59E0B !important; }
.stError   { border-left: 3px solid #FF4444 !important; }

/* ── Progress / Spinner ──────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div { background-color: #2563EB !important; }
[data-testid="stSpinner"] > div { border-top-color: #2563EB !important; }

/* ── Divider / Caption / Code ────────────────────────────────────────────── */
hr { border-color: #1E1E2E !important; margin: 1.25rem 0 !important; }
small, .stCaption, [data-testid="stCaptionContainer"] p {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
}
code { background-color: #1E1E2E !important; color: #60A5FA !important; border-radius: 4px !important; }
.stCode > div {
    background-color: #1E1E2E !important;
    border: 1px solid #2E2E3E !important;
    border-radius: 8px !important;
}

/* ── Plotly chart ────────────────────────────────────────────────────────── */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0A0F; }
::-webkit-scrollbar-thumb { background: #1E1E2E; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2563EB; }
"""


def inject_css() -> None:
    """Inject dark fintech theme CSS. Call once at the top of every page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------

def apply_chart_theme(fig, title: str = "") -> object:
    """Apply the dark theme to any Plotly figure. Returns the figure."""
    fig.update_layout(
        paper_bgcolor="#111118",
        plot_bgcolor="#111118",
        font=dict(color="#94A3B8", family="Inter, sans-serif", size=12),
        title=dict(text=title, font=dict(color="#F1F5F9", size=15, family="Inter"), x=0.01),
        xaxis=dict(
            gridcolor="#1E1E2E", linecolor="#1E1E2E",
            tickcolor="#94A3B8", tickfont=dict(color="#94A3B8"),
            zerolinecolor="#1E1E2E",
        ),
        yaxis=dict(
            gridcolor="#1E1E2E", linecolor="#1E1E2E",
            tickcolor="#94A3B8", tickfont=dict(color="#94A3B8"),
            zerolinecolor="#1E1E2E",
        ),
        legend=dict(
            bgcolor="rgba(17,17,24,0.85)", bordercolor="#1E1E2E", borderwidth=1,
            font=dict(color="#94A3B8", size=11),
        ),
        margin=dict(l=50, r=20, t=45, b=40),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Reusable HTML components
# ---------------------------------------------------------------------------

def card(html: str, padding: str = "1.25rem 1.4rem") -> None:
    """Render arbitrary HTML inside a dark card."""
    st.markdown(
        f'<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;'
        f'padding:{padding};">{html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    subtitle: str = "",
    value_color: str = "#F1F5F9",
) -> None:
    sub = (
        f'<p style="color:#94A3B8;font-size:0.78rem;margin:0.3rem 0 0;">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="background:#111118;border:1px solid #1E1E2E;border-radius:8px;padding:1rem 1.25rem;">'
        f'<p style="color:#94A3B8;font-size:0.72rem;font-weight:600;letter-spacing:0.07em;'
        f'text-transform:uppercase;margin:0 0 0.4rem 0;">{label}</p>'
        f'<p style="color:{value_color};font-size:1.65rem;font-weight:700;margin:0;">{value}</p>'
        f'{sub}'
        f'</div>',
        unsafe_allow_html=True,
    )


def signal_card(label: str, signal: str, subtitle: str = "", is_bullish: bool = True) -> None:
    color = THEME["bullish"] if is_bullish else THEME["bearish"]
    arrow = "▲" if is_bullish else "▼"
    sub = (
        f'<p style="color:#94A3B8;font-size:0.82rem;margin:0.5rem 0 0;">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="background:#111118;border:1px solid {color}33;border-radius:8px;padding:1.1rem 1.25rem;">'
        f'<p style="color:#94A3B8;font-size:0.72rem;font-weight:600;letter-spacing:0.07em;'
        f'text-transform:uppercase;margin:0 0 0.5rem 0;">{label}</p>'
        f'<p style="color:{color};font-size:1.55rem;font-weight:700;margin:0;">{arrow} {signal}</p>'
        f'{sub}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<p style="color:#94A3B8;font-size:0.85rem;margin:0.2rem 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div style="margin:1.5rem 0 0.75rem;">'
        f'<h3 style="color:#F1F5F9;font-size:1.1rem;font-weight:700;margin:0;">{title}</h3>'
        f'{sub}</div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94A3B8;font-size:0.75rem;text-align:center;">'
        "Not financial advice — for educational and research purposes only.</p>",
        unsafe_allow_html=True,
    )
