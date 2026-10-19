"""Public identity and navigation for the Gradio interface; no engine imports."""

import html
import json
from pathlib import Path
from urllib.parse import urljoin

_ASSETS = Path(__file__).resolve().parent
ECOSYSTEM = json.loads((_ASSETS / "ecosystem.json").read_text(encoding="utf-8"))


def guide_url(guide):
    return urljoin(ECOSYSTEM["links"]["docs"], ECOSYSTEM["guides"][guide])


def guide_markdown(guide, translate=lambda text: text):
    return f'[{translate("User guide")}]({guide_url(guide)})'


def resource_markdown(translate=lambda text: text):
    links = ECOSYSTEM["links"]
    destinations = [
        ("Website", links["website"]),
        ("Download Desktop", links["download"]),
        ("Documentation", links["docs"]),
        ("Community", links["community"]),
        ("Support us", links["support"]),
        ("Source code", ECOSYSTEM["projects"]["gradio"]["repository"]),
    ]
    if links.get("patreon"):
        destinations.insert(5, ("Patreon", links["patreon"]))
    return " · ".join(f"[{translate(label)}]({url})" for label, url in destinations)


def brand_header():
    mark = (_ASSETS / "applio-logo.svg").read_text(encoding="utf-8")
    mark = mark.replace("#EAEAEA", "currentColor").replace(
        "<svg ",
        '<svg aria-hidden="true" style="width:24px;height:28px;flex:none" ',
        1,
    )
    name = html.escape(ECOSYSTEM["brand"]["name"])
    return (
        '<div style="display:flex;align-items:center;gap:10px">'
        f'{mark}<h1 style="margin:0">{name}</h1>'
        '<span style="font-size:12px;opacity:.7">Gradio</span></div>'
    )
