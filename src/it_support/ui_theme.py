"""Presentation constants and stylesheet for the Streamlit interface."""

from __future__ import annotations

STATUS_TONE = {
    "Open": ("#b42318", "#fef3f2"),
    "In Progress": ("#b54708", "#fffaeb"),
    "Resolved": ("#067647", "#ecfdf3"),
    "Closed": ("#475467", "#f2f4f7"),
}

PRIORITY_TONE = {
    "Critical": ("#b42318", "#fef3f2"),
    "High": ("#b54708", "#fffaeb"),
    "Medium": ("#175cd3", "#eff8ff"),
    "Low": ("#475467", "#f2f4f7"),
}

SERVICE_TONE = {
    "Operational": ("#067647", "#ecfdf3"),
    "Degraded": ("#b54708", "#fffaeb"),
    "Outage": ("#b42318", "#fef3f2"),
}

STYLESHEET = """
<style>
:root {
  --ink: #101828;
  --muted: #667085;
  --line: #e4e7ec;
  --surface: #ffffff;
  --surface-sunk: #f9fafb;
  --accent: #175cd3;
  --rail-w: 390px;
}

/* The palette is light by design; pin the canvas to it so the app never ends up
   with light cards floating on a dark browser/OS theme. */
html, body, .stApp,
section[data-testid="stMain"],
div[data-testid="stAppViewContainer"] {
  background: var(--surface-sunk) !important;
  color: var(--ink) !important;
  color-scheme: light;
}
section[data-testid="stSidebar"] > div { background: var(--surface) !important; }
section[data-testid="stSidebar"] * { color: var(--ink); }

/* No auto-centering: the content spans the gap between the two sidebars, with
   a matching gutter on each side. Centering inside a padded main left a wide
   dead strip against the left sidebar. */
.block-container {
  max-width: none !important;
  margin: 0 !important;
  padding: 1.6rem 2.5rem 6rem;
  padding-right: calc(var(--rail-w) + 2.5rem);
}

/* Both columns start at the same y and keep a common gutter. */
div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
div[data-testid="stColumn"] { min-width: 0; }

/* Tables sized themselves before layout settled and got clipped mid-column. */
div[data-testid="stDataFrame"] { width: 100% !important; max-width: 100% !important; }
div[data-testid="stDataFrame"] > div { width: 100% !important; }

/* The docked chat input tracks the same gutters as the content above it. */
div[data-testid="stBottomBlockContainer"] {
  max-width: none !important; margin: 0 !important;
  padding: .6rem 2.5rem 1.1rem;
  padding-right: calc(var(--rail-w) + 2.5rem);
}

/* Widgets otherwise inherit the browser theme. */
.stChatMessage, div[data-testid="stExpander"] details,
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  background: var(--surface) !important;
  color: var(--ink) !important;
}
div[data-testid="stChatInput"] {
  background: var(--surface) !important;
  border: 1px solid var(--line); border-radius: 12px;
}
div[data-testid="stChatInput"] textarea { color: var(--ink) !important; }
div[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
div[data-testid="stBottomBlockContainer"] { background: var(--surface-sunk) !important; }
.stButton button {
  background: var(--surface); color: var(--ink);
  border: 1px solid var(--line); border-radius: 10px;
  text-align: left; font-size: .84rem; line-height: 1.35;
}
.stButton button:hover { border-color: var(--accent); color: var(--accent); }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent); }

/* --- header ---------------------------------------------------------- */
.app-header {
  display: flex; align-items: center; gap: .9rem;
  padding: 1rem 1.25rem; margin-bottom: 1rem;
  border: 1px solid var(--line); border-radius: 14px;
  background: linear-gradient(120deg, #f8fafc 0%, #eef4ff 100%);
}
.app-header .glyph { font-size: 1.9rem; line-height: 1; }
.app-header h1 { margin: 0; font-size: 1.35rem; letter-spacing: -.01em; color: var(--ink); }
.app-header p { margin: .18rem 0 0; font-size: .85rem; color: var(--muted); }

/* --- generic chips ---------------------------------------------------- */
.chip {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .16rem .55rem; border-radius: 999px;
  font-size: .74rem; font-weight: 600; letter-spacing: .01em;
  white-space: nowrap;
}

/* --- identified employee banner --------------------------------------- */
.identity {
  display: flex; align-items: center; gap: .85rem;
  padding: .7rem .9rem; margin-bottom: .85rem;
  border: 1px solid #b2ddff; border-left: 4px solid var(--accent);
  border-radius: 12px; background: #eff8ff;
}
.identity .avatar {
  width: 38px; height: 38px; flex: 0 0 38px;
  display: grid; place-items: center; border-radius: 50%;
  background: var(--accent); color: #fff; font-weight: 700; font-size: .85rem;
}
.identity .who { font-weight: 700; color: var(--ink); font-size: .92rem; }
.identity .meta { font-size: .78rem; color: var(--muted); }
.identity .empid {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--accent); color: #fff;
  padding: .1rem .45rem; border-radius: 6px; font-size: .78rem;
}
.identity.anonymous {
  border-color: var(--line); border-left-color: #98a2b3; background: var(--surface-sunk);
}
.identity.anonymous .avatar { background: #98a2b3; }

/* highlight an employee ID wherever it appears in an answer */
.empid-inline {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #eff8ff; color: #175cd3; border: 1px solid #b2ddff;
  padding: 0 .3rem; border-radius: 5px; font-weight: 600;
}

/* --- reference panel -------------------------------------------------- */
.panel-title {
  font-size: .78rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted); margin: .2rem 0 .5rem;
}
.kb-card, .ticket-card {
  border: 1px solid var(--line); border-radius: 11px;
  padding: .6rem .75rem; margin-bottom: .5rem; background: var(--surface);
}
.kb-card .kb-title { font-weight: 650; font-size: .87rem; color: var(--ink); }
.kb-card .kb-meta  { font-size: .73rem; color: var(--muted); margin-top: .15rem; }
.ticket-card .row  { display: flex; justify-content: space-between; gap: .5rem; align-items: center; }
.ticket-card .tid  {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .78rem; font-weight: 650; color: var(--ink);
}
.ticket-card .subject { font-size: .84rem; color: var(--ink); margin-top: .25rem; }
.ticket-card .meta    { font-size: .73rem; color: var(--muted); margin-top: .2rem; }

/* --- counters --------------------------------------------------------- */
.counters { display: flex; gap: .4rem; flex-wrap: wrap; margin-bottom: .6rem; }
.counter {
  flex: 1 1 0; min-width: 74px; text-align: center;
  border: 1px solid var(--line); border-radius: 10px;
  padding: .4rem .2rem; background: var(--surface);
}
.counter .n { font-size: 1.1rem; font-weight: 700; color: var(--ink); line-height: 1.1; }
.counter .l { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }

/* --- misc ------------------------------------------------------------- */
div[data-testid="stExpander"] details { border-radius: 10px; border-color: var(--line); }
.stChatMessage {
  border-radius: 12px; border: 1px solid var(--line);
  padding: .7rem .9rem; margin-bottom: .6rem;
}
section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }

/* --- right sidebar ---------------------------------------------------- */
/* Streamlit only ships a left sidebar, so the reference column is lifted out
   of the layout flow and pinned to the right edge, mirroring it. The
   `.rail-anchor` div is emitted at the top of that column; matching on it
   avoids catching the nested columns inside the panels. */
div[data-testid="stColumn"]:has(.rail-anchor) {
  position: fixed; top: 0; right: 0; bottom: 0;
  width: var(--rail-w) !important; flex: none !important;
  z-index: 5;                       /* under Streamlit's own header toolbar */
  background: var(--surface);
  border-left: 1px solid var(--line);
  padding: 3.6rem 1.15rem 1.5rem;
  overflow-y: auto; overscroll-behavior: contain;
}
div[data-testid="stColumn"]:has(.rail-anchor)::-webkit-scrollbar { width: 8px; }
div[data-testid="stColumn"]:has(.rail-anchor)::-webkit-scrollbar-thumb {
  background: #d0d5dd; border-radius: 8px;
}
div[data-testid="stColumn"]:has(.rail-anchor) .stTabs [data-baseweb="tab"] {
  font-size: .82rem; padding: .3rem .45rem;
}
div[data-testid="stColumn"]:has(.rail-anchor) .kb-card,
div[data-testid="stColumn"]:has(.rail-anchor) .counter {
  background: var(--surface-sunk);
}

/* With the rail out of flow, the conversation column takes the whole row. */
div[data-testid="stHorizontalBlock"]:has(.rail-anchor) > div[data-testid="stColumn"]:first-child {
  flex: 1 1 100% !important; width: 100% !important;
}

@media (max-width: 1200px) {
  /* Too narrow for a second sidebar: return the panel to the page flow. */
  div[data-testid="stColumn"]:has(.rail-anchor) {
    position: static; width: 100% !important; flex: 1 1 100% !important;
    border-left: none; border-top: 1px solid var(--line);
    padding: 1rem 0 0; overflow: visible;
  }
  .block-container { padding-right: 2.5rem; }
  div[data-testid="stBottomBlockContainer"] { padding-right: 2.5rem; }
}

</style>
"""


def chip(text: str, tone: tuple[str, str]) -> str:
    """Render a coloured pill."""
    colour, background = tone
    return (
        f'<span class="chip" style="color:{colour};background:{background};'
        f'border:1px solid {colour}22">{text}</span>'
    )


def status_chip(status: str) -> str:
    return chip(status, STATUS_TONE.get(status, STATUS_TONE["Closed"]))


def priority_chip(priority: str) -> str:
    return chip(priority, PRIORITY_TONE.get(priority, PRIORITY_TONE["Low"]))


def service_chip(status: str) -> str:
    return chip(status, SERVICE_TONE.get(status, SERVICE_TONE["Operational"]))
