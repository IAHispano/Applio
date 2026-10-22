"""Gradio adaptation of the Desktop sidebar, workspace and design tokens.

Reference: frontend/web/components/layout/{Sidebar,nav} and app/globals.css.
The engine's existing page builders remain mounted when navigating.
"""

import base64
from contextlib import contextmanager
from functools import partial
import html
import json
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from assets.app_links import ECOSYSTEM, guide_url

_ROOT = Path(__file__).resolve().parent / "ui"

# Navigation follows the app, exposing the tools available in the Gradio engine.
SECTIONS = (
    (
        "Voice",
        (
            ("inference", "Convert Audio"),
            ("realtime", "Realtime"),
            ("tts", "Text to Speech"),
        ),
    ),
    (
        "Models & Training",
        (
            ("models", "Models"),
            ("training", "Training"),
            ("blender", "Voice Blender"),
            ("tensorboard", "TensorBoard"),
        ),
    ),
    ("Tools & Extras", (("audio", "Audio Tools"), ("plugins", "Plugins"))),
    (
        "System",
        (
            ("terminal", "Terminal"),
            ("settings", "Settings"),
            ("report", "Report a Bug"),
        ),
    ),
)

DESCRIPTIONS = {
    "inference": "Convert audio files using trained voice models with single and batch processing.",
    "realtime": "Transform your voice live with microphone input and realtime controls.",
    "tts": "Generate speech from text and convert it with your voice models.",
    "models": "Download and import voice models, indexes and pretrained weights.",
    "training": "Prepare your dataset, extract features, train and export a voice model.",
    "blender": "Combine two voice models into a single checkpoint.",
    "tensorboard": "Monitor training loss and metrics in real time.",
    "audio": "Inspect models, analyze audio and extract pitch curves.",
    "plugins": "Manage extensions for your Applio workspace.",
    "terminal": "Follow live application output, progress and errors without leaving your workspace.",
    "settings": "Configure your engine, audio devices and preferences.",
    "report": "Inspect system information and get help with an issue.",
}

# Exact SVG nodes from the app's lucide-react 1.47.0 package (ISC).
# Shared entries follow frontend/web/components/layout/nav.ts; Terminal is also
# used by TrainingConsole.tsx. See ui/LUCIDE-LICENSE.md for attribution.
_ICONS = json.loads((_ROOT / "icons.json").read_text(encoding="utf-8"))


def stylesheet():
    css = (_ROOT / "shell.css").read_text(encoding="utf-8")
    font = _ROOT / "Syne-Variable.ttf"
    if font.is_file():
        data = base64.b64encode(font.read_bytes()).decode("ascii")
        css += (
            '\n@font-face{font-family:"Applio Syne";font-style:normal;font-weight:400 800;font-display:swap;src:url(data:font/ttf;base64,'
            + data
            + ') format("truetype");}'
        )
    for key, svg in _ICONS.items():
        css += f'\n.applio-nav-{key}::before,.applio-link-{key}::before{{mask-image:url("data:image/svg+xml,{quote(svg)}");}}'
    return css


def create_theme():
    return gr.themes.Base(
        primary_hue="neutral",
        secondary_hue="neutral",
        neutral_hue="neutral",
        font=[
            gr.themes.Font(name) for name in ("Applio Syne", "Segoe UI", "sans-serif")
        ],
        font_mono=[gr.themes.Font(name) for name in ("Consolas", "monospace")],
    )


def _link(label, url, icon=None):
    classes = f' class="applio-footer-link applio-link-{icon}"' if icon else ""
    return f'<a{classes} href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(label)}</a>'


def page_header(key, title, translate):
    guide = ECOSYSTEM["guides"].get(key)
    help_link = _link(translate("User guide") + " ↗", guide_url(key)) if guide else ""
    gr.HTML(
        '<header class="applio-page-heading">'
        f"<div><h1>{html.escape(translate(title))}</h1>"
        f"<p>{html.escape(translate(DESCRIPTIONS[key]))}</p></div>{help_link}</header>",
        elem_classes="applio-page-header",
    )


def step_header(number, title, translate=lambda text: text, description=None):
    desc = (
        f'<p class="applio-card-desc">{html.escape(translate(description))}</p>'
        if description
        else ""
    )
    gr.HTML(
        '<div class="applio-card-head">'
        '<h2 class="applio-card-heading">'
        f'<span class="applio-step">{number}</span>{html.escape(translate(title))}</h2>'
        f"{desc}</div>",
        elem_classes="applio-step-header",
    )


@contextmanager
def step_card(number, title, translate=lambda text: text, description=None):
    with gr.Column(scale=0, min_width=0, elem_classes="applio-card"):
        step_header(number, title, translate, description)
        yield


def build_workspace(builders, translate=lambda text: text):
    """Build persistent Gradio workspaces with app-style sidebar navigation."""
    nav = {}
    entries = [item for _, items in SECTIONS for item in items]
    with gr.Row(elem_id="applio-shell", equal_height=False):
        with gr.Column(scale=0, min_width=220, elem_id="applio-sidebar"):
            version = json.loads(
                (_ROOT.parent / "config_template.json").read_text(encoding="utf-8")
            )["version"]
            mark = (_ROOT.parent / "applio-logo.svg").read_text(encoding="utf-8")
            mark = mark.replace("#EAEAEA", "currentColor").replace(
                "<svg ", '<svg aria-hidden="true" ', 1
            )
            gr.HTML(
                f'<div class="applio-brand">{mark}<span>Applio</span>'
                f'<small title="Gradio interface">v{html.escape(str(version))}</small></div>'
            )
            menu = gr.Button(translate("Menu"), size="sm", elem_id="applio-menu-toggle")
            menu.click(
                fn=None,
                js="""() => {
                    const sidebar = document.getElementById('applio-sidebar');
                    const open = sidebar.classList.toggle('mobile-open');
                    const button = document.getElementById('applio-menu-toggle');
                    button.setAttribute('aria-expanded', String(open));
                    button.setAttribute('aria-controls', 'applio-navigation');
                }""",
                queue=False,
            )
            with gr.Column(elem_id="applio-navigation"):
                for section, items in SECTIONS:
                    with gr.Column(
                        scale=0, min_width=0, elem_classes="applio-nav-group"
                    ):
                        if section:
                            gr.HTML(
                                f"<h2>{html.escape(translate(section))}</h2>",
                                elem_classes="applio-nav-heading",
                            )
                        for key, label in items:
                            nav[key] = gr.Button(
                                translate(label),
                                variant=(
                                    "primary" if key == "inference" else "secondary"
                                ),
                                elem_classes=["applio-nav-item", f"applio-nav-{key}"],
                            )
            links = ECOSYSTEM["links"]
            gr.HTML(
                '<div class="applio-sidebar-footer">'
                + '<div class="applio-sidebar-resources">'
                + _link(translate("Website"), links["website"], "website")
                + _link(translate("Documentation"), links["docs"], "docs")
                + "</div>"
                + _link(translate("Discord support"), links["discord"], "community")
                + '<div class="applio-sidebar-support">'
                + _link(translate("Support us"), links["support"], "support")
                + '<span aria-hidden="true">·</span>'
                + _link(translate("Patreon"), links["patreon"])
                + "</div>"
                + "</div>"
            )

        with gr.Column(elem_id="applio-content", min_width=0):
            with gr.Tabs(selected="inference", elem_id="applio-pages") as pages:
                for key, title in entries:
                    with gr.Tab(
                        title,
                        id=key,
                        elem_id=f"applio-page-{key}",
                        elem_classes="applio-page",
                    ):
                        page_header(key, title, translate)
                        builders[key]()
            gr.HTML(
                '<footer class="applio-footer">'
                + _link(
                    translate("Terms of use"),
                    "https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md",
                )
                + "</footer>"
            )

    def navigate(key):
        return [gr.Tabs(selected=key)] + [
            gr.Button(variant="primary" if item == key else "secondary") for item in nav
        ]

    def bind_navigation(button, key):
        event = button.click(
            partial(navigate, key),
            outputs=[pages, *nav.values()],
            queue=False,
            show_progress="hidden",
        )
        event.then(
            fn=None,
            js="""() => requestAnimationFrame(() => requestAnimationFrame(() => {
                window.dispatchEvent(new Event('resize'));
                const page = [...document.querySelectorAll('#applio-pages > [role="tabpanel"]')]
                    .find(node => node.getBoundingClientRect().width > 0);
                const heading = page?.querySelector('h1');
                if (heading) { heading.tabIndex = -1; heading.focus({preventScroll: true}); }
                document.getElementById('applio-sidebar')?.classList.remove('mobile-open');
                document.getElementById('applio-menu-toggle')?.setAttribute('aria-expanded', 'false');
                document.querySelectorAll('#applio-navigation button').forEach(button => {
                    if (button.classList.contains('primary')) button.setAttribute('aria-current', 'page');
                    else button.removeAttribute('aria-current');
                });
                window.scrollTo({top: 0, behavior: 'instant'});
            }))""",
            queue=False,
        )

    for key, button in nav.items():
        bind_navigation(button, key)
    return pages
