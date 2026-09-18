# Applio Theme System

Applio (`app/web`) themes are plain JSON files in `assets/themes/`. This replaces the
Gradio theme picker (`assets/themes/*.py` + Hub IDs in `theme_list.json`) from
`/home/vidalnt/Applio-main/`, which cannot work outside Gradio: Hub themes are Python
classes resolved at `launch()` time. Ours apply **live in the browser** — no restart,
no rebuild.

## 30-second start

1. Copy the template:
   `cp assets/themes/custom.example.json assets/themes/my-theme.json`
2. Edit any value (all keys optional — missing keys fall back to the built-in default,
   so a theme can be a single accent color).
3. Open **Settings → Appearance**, pick your file, press **Save**. It applies instantly
   and is remembered in `assets/config.json` (`theme.file`, `""` = default).

## File format

```jsonc
{
  "name": "My Theme",          // shown in the selector
  "version": "1.0.0",          // informational (Gradio push_to_hub parity: semver)
  "description": "...",        // shown truncated in the selector
  "mode": "dark",              // informational; exposed as data-theme-mode
  "colors": { "primary": "#7c3aed", ... },
  "fonts": {
    "display": ["google:Space Grotesk", "sans-serif"],
    "body": ["Inter", "system-ui", "sans-serif"],
    "mono": ["ui-monospace", "monospace"]
  },
  "radius": { "card": "16px", "input": "10px", "button": "999px", "pill": "999px" },
  "shadows": { "card": "0 8px 30px rgba(0,0,0,.45)", "button": "none" },
  "custom_css": ".card { border-width: 2px; }"
}
```

### `colors` — full token list

| Key | Controls | Default |
|---|---|---|
| `primary` | Accent: CTA buttons, sliders, scrollbars, active states | `#ffffff` |
| `primarySoft` | Soft accent surfaces | `rgba(255,255,255,.12)` |
| `background` | Page background | `#0a0a0a` |
| `surface` | Cards | `rgba(255,255,255,.05)` |
| `surfaceDim` | Inputs, checkboxes | `rgba(0,0,0,.4)` |
| `panel` | Dropdown popup | `#1c1c1c` |
| `border` | Borders (cards, inputs, badges) | `rgba(255,255,255,.1)` |
| `text` | Body text | `#e7e5e4` |
| `heading` | Headings, hover text | `#f5f5f5` |
| `muted` | Labels, hints | `#a3a3a3` |
| `ctaBg` / `ctaText` | Primary buttons | `#ffffff` / `#0a0a0a` |
| `buttonBg` / `buttonBgHover` / `buttonTextHover` | Default buttons + hover | `rgba(255,255,255,.08/.14)`, `#ffffff` |
| `buttonGhostText` / `buttonGhostBorder` / `buttonGhostBorderHover` | Ghost buttons | `#d4d4d4`, `.14`, `.28` white |
| `sliderTrack` / `sliderThumb` | Range inputs | `.12` white / `#ffffff` |
| `checkboxBorder` / `checkboxBorderHover` / `checkboxChecked` / `checkboxCheck` | Checkboxes (radios reuse these tokens) | `.25` / `.5` white, `#fff`, `#000` |
| `fileButtonBg` / `fileButtonBorder` / `fileButtonText` / `fileButtonBgHover` | File-picker button (`::file-selector-button`) | transparent, `.14` white, `#d4d4d4`, `.08` white |
| `focusBorder` | Focus rings | `rgba(255,255,255,.25)` |
| `selectionBg` / `selectionText` | Text selection | `#fff` / `#000` |
| `logBg` | Log/terminal blocks | `rgba(0,0,0,.5)` |
| `ok` / `warn` / `err` | Status badges | `#4ade80` / `#fbbf24` / `#f87171` |
| `dangerBg` / `dangerBorder` / `dangerText` (+ `*Hover`) | Destructive buttons | red trio (+ hover trio) |

Any value accepts what CSS accepts — including **gradients**:
`"ctaBg": "linear-gradient(90deg, #7c3aed, #2563eb)"`. Unknown keys are ignored, so
forward-compatible tokens never break old files.

### `fonts`

Stacks like Gradio's `font=[...]` lists. Prefix a family with `google:` to load it
from Google Fonts on demand (Gradio `GoogleFont` parity):
`["google:Space Grotesk", "sans-serif"]`. `display` styles titles, `body` the app
text, `mono` the log blocks.

### `radius` / `shadows` / `titlebar`

`card` (cards), `input` (inputs, logs, details), `button` (buttons),
`pill` (sliders, badges). `shadows.card` / `shadows.button` take any `box-shadow`.
`titlebar.bg` tints the desktop window chrome, `titlebar.closeBg`/`closeText` the
close-button hover (platform red by default).

### `custom_css`

Raw CSS appended after every token (Gradio `custom_css` parity) — for anything
tokens cannot express. Example:

```css
.card { border-width: 2px; }
button.cta { text-transform: uppercase; letter-spacing: .05em; }
```

## Where this is more expressive than Gradio

Gradio themes revolve around **8 core variables** (3 hue palettes + 3 sizes + 2 fonts);
everything else derives from palettes, and direct control needs `.set()` with
`*` references. Here every token takes a **final value** (no palette indirection),
states have first-class keys (`*Hover`), any CSS value works (gradients, shadows),
and `custom_css` ships inside the same file. A one-accent theme is 5 lines; a total
restyle never leaves the file.

## Gradio mapping cheat-sheet

| Gradio (`theming-guide`) | Ours |
|---|---|
| `primary_hue` / `secondary_hue` / `neutral_hue` | `colors.primary` (+ direct tokens) |
| `spacing_size` / `radius_size` / `text_size` | `radius.*` (per element) |
| `font` / `font_mono` (+ `GoogleFont`) | `fonts.display/body/mono` (+ `google:`) |
| `theme.set(loader_color=…)` | any `colors.*` key |
| `theme.custom_css` | `custom_css` |
| `theme.dump("seafoam.json")` / Hub share | copy the JSON / send the file |
| `theme_list.json` Hub IDs | `assets/themes/*.json` (local, instant) |

The original `Applio.py` theme is ported 1:1 as `assets/themes/applio.json`
(warm `#110F0F` background, neutral grays, blue `#2563eb` focus/checkbox accents,
Syne + Nunito Sans). Select it in Settings → Appearance to get the Gradio look back.

## API

- `GET /api/settings/themes` → `{ themes: [{ id, name, description, example }], selected }`
- `GET /api/settings/theme[?file=]` → `{ id, theme }` (`""` = built-in default)
- `PUT /api/settings` with `{ "theme": { "file": "my-theme.json" } }` persists the
  choice; the UI then fires `applio:theme-changed` and `ThemeProvider` re-applies
  without reload.

## Troubleshooting

- **Theme not listed**: must be valid JSON directly under `assets/themes/` (no
  subfolders). Invalid files are listed with an error description instead.
- **Nothing changes on Save**: open devtools — unknown keys are ignored by design;
  check key spelling against the table above.
- **Google font doesn't load**: offline builds can't reach `fonts.googleapis.com`;
  fallbacks in the stack apply automatically.
- **Want light mode?** There is no separate light stylesheet: set light values for
  `background/surface/text/...` and `"mode": "light"`. Some raster assets
  (slider SVGs, noise) assume dark and stay as-is.
