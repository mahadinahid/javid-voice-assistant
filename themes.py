"""Colour themes: palettes on the Glass look, one small JSON file each in themes/ (2026-10-26; palette roles, light
backgrounds and the Hifz page's colours 2026-11-23).

A theme file gives a name, a background, an accent, optionally three glow colours, and the glass's blur and tint. A
palette may also name its own roles: the panel, the raised panel (menus and dialogs), the border, the text, the muted
text and the text on the accent; whatever it leaves out is worked out. derive() works out every other colour the page
uses and checks that text stays readable: at least 4.5:1 wherever it appears, measured at the glows' strongest and
through the glass's tint, as Glass itself was checked. A theme that can't be made readable is refused, so no theme can
leave Javid unreadable. The page sets the derived colours as CSS variables on <html>; the looks themselves (Orbit,
Glass, Overdrive) stay CSS in index.html. A file with none of the roles is worked out exactly as before they existed.

A background is dark (up to MAX_BG_LUMINANCE, about #303030) or light (from MIN_LIGHT_LUMINANCE, about #C4C4C4). A
light theme gets dark text, dark tints where the dark themes have white ones, the browser's light controls and softer
shadows (index.html reads them as --ink-rgb, --field-rgb and html[data-tone="light"]). A background in between is
refused: neither the dark glass's white edges nor the light glass's dark ones show on a mid-grey.

hifz() maps a theme onto the Hifz page's own roles (its background and star pattern, the cards, the Arabic text and
the translation, the accent, the player) the way Hifz Classic, the page's own green and gold, maps them. The six juz
status colours stay Classic's set, moved together in lightness only as far as contrast asks, and hifz_checks() checks
the Arabic text, the ayah highlight and every status, and that the six stay apart (CIEDE2000, a standard measure of
how different two colours look).
"""

import colorsys
import itertools
import json
import math
import os
import re
import tempfile

THEMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")
RESERVED = ("orbit", "glass", "aurora", "overdrive", "classic", "preview")   # the looks in index.html, the names older
                                                                     # versions saved, and the editor's route
PRESETS = ("ember", "graphite", "ocean", "orchid", "slate", "royal-blue", "plum", "mist", "teal-rust", "berry-coral",
           "violet", "navy-amber", "coffee", "olive")       # the files Javid ships: the Hifz page offers these
ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")
NAME_MAX = 24
MAX_THEMES = 40
MAX_FILE_BYTES = 16_000
BLUR = (0, 28, 18)                 # px: lowest, highest, default (Glass's)
TINT = (0.40, 0.85, 0.58)          # the glass's tint opacity: lowest, highest, default (Glass's)
MAX_BG_LUMINANCE = 0.03            # about #303030: the lightest a dark background may be
MIN_LIGHT_LUMINANCE = 0.55         # about #C4C4C4: the darkest a light one may be
MIN_TEXT = 4.5                     # WCAG AA for text
MIN_BUTTONS = 3.0                  # a button against the glass it sits on (WCAG's 3:1 for controls)
GLOW_ALPHA = (0.24, 0.16, 0.09)    # top left, bottom right, top centre: Glass's
ROLES = (("panel", "The panel"), ("raised", "The raised panel"), ("border", "The border"), ("text", "The text"),
         ("muted", "The muted text"), ("on_accent", "The text on the accent"))
FIELDS = ("name", "background", "panel", "raised", "border", "text", "muted", "accent", "on_accent", "glows", "blur",
          "tint")                  # a file's keys, in the order they're written
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
OK_GREEN = (74, 222, 128)          # "up" and "serving": the same green in every colour theme
STATUS_COLOURS = {"--amber": (255, 190, 85), "--red": (255, 107, 129), "--pink": (251, 113, 133), "--mint": OK_GREEN}
STATUS_TEXTS = {"--ok-text": (143, 245, 208), "--warn-text": (255, 211, 138), "--warn-text-2": (255, 154, 107),
                "--bad-text": (255, 156, 171)}    # the pastels the page writes these in on Glass (its fallbacks)
PILL_ALPHA, TOGGLE_ALPHA = 0.06, 0.14     # Glass's pill fill, and the selected language button's
FADES = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)   # how far a faded item may go (Glass: 0.6 to 0.8)
LIGHT_SHEEN = (0.30, 0.10)         # a light theme's glass: its white sheen at the top and the bottom
ROLE_LINES = {"--glass-border": 0.55, "--line": 0.45, "--line-strong": 0.75, "--pill-border": 0.6}   # the border role
EDGE_LINES = {"--glass-border": 0.10, "--line": 0.08, "--line-strong": 0.14, "--pill-border": 0.12}  # Glass's, in ink


class ThemeError(ValueError):
    """A theme that can't be used, said so the page can show it."""


# ---------- colour arithmetic (sRGB 0-255 tuples) ----------
def _rgb(hex_colour: str) -> tuple:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def _hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02X}" for c in rgb)


def _triple(rgb) -> str:
    return ", ".join(str(max(0, min(255, round(c)))) for c in rgb)


def _rgba(rgb, alpha: float) -> str:
    return f"rgba({_triple(rgb)}, {alpha:g})"


def _lum(rgb) -> float:
    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a, b) -> float:
    high, low = sorted((_lum(a), _lum(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _over(top, alpha: float, base) -> tuple:
    """`top` at `alpha` painted over an opaque `base`."""
    return tuple(t * alpha + b * (1 - alpha) for t, b in zip(top, base))


def _mix(a, b, share: float) -> tuple:
    return _over(b, share, a)


def _hls(rgb) -> tuple:
    return colorsys.rgb_to_hls(*(c / 255 for c in rgb))


def _from_hls(hue: float, light: float, sat: float) -> tuple:
    return tuple(c * 255 for c in colorsys.hls_to_rgb(hue % 1.0, max(0.0, min(1.0, light)), max(0.0, min(1.0, sat))))


def _shift(rgb, degrees: float) -> tuple:
    hue, light, sat = _hls(rgb)
    return _from_hls(hue + degrees / 360, light, sat)


def _readable(rgb, surfaces, minimum: float = MIN_TEXT + 0.1, darken: bool = False) -> tuple:
    """`rgb`, lightened in its own hue until it reads at `minimum` on every surface (they're all dark), or darkened
    for a light theme's surfaces."""
    hue, light, sat = _hls(rgb)
    colour = rgb
    step = -0.01 if darken else 0.01
    while (light > 0.015 if darken else light < 0.985) and min(contrast(colour, s) for s in surfaces) < minimum:
        light += step
        colour = _from_hls(hue, light, sat)
    return colour


def _glows_for(accent) -> list:
    """Glows that match the accent, as Glass's do its own: a deeper shade, a neighbour on each side."""
    return [_mix(accent, BLACK, 0.2), _shift(accent, 30), _shift(accent, -45)]


# ---------- CIELAB (D65), for moving a colour in lightness alone and for how far apart two colours look ----------
def _linear(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lab(rgb) -> tuple:
    r, g, b = (_linear(c) for c in rgb)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _lab_linear(L: float, a: float, b: float) -> tuple:
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: t ** 3 if t ** 3 > 216 / 24389 else (116 * t - 16) / (24389 / 27)
    x, y, z = inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883
    return (3.2404542 * x - 1.5371385 * y - 0.4985314 * z, -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
            0.0556434 * x - 0.2040259 * y + 1.0572252 * z)


def _from_lab(L: float, a: float, b: float) -> tuple:
    enc = lambda c: 12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0) ** (1 / 2.4) - 0.055
    return tuple(max(0.0, min(255.0, enc(c) * 255)) for c in _lab_linear(L, a, b))


def _at(colour, dl: float) -> tuple:
    """`colour` made `dl` lighter in CIELAB (darker if negative), its hue kept; its chroma only as far reduced as
    the screen's colours need."""
    L, a, b = _lab(colour)
    L = max(0.0, min(100.0, L + dl))
    k = 1.0
    while k > 0 and not all(-0.001 <= c <= 1.001 for c in _lab_linear(L, a * k, b * k)):
        k -= 0.02
    return _from_lab(L, a * max(k, 0), b * max(k, 0))


def delta_e(c1, c2) -> float:
    """CIEDE2000: about 1 is the smallest difference an eye sees side by side; 10 and more is a different colour."""
    L1, a1, b1 = _lab(c1)
    L2, a2, b2 = _lab(c2)
    mean_c = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(mean_c ** 7 / (mean_c ** 7 + 25 ** 7)))
    a1, a2 = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1, b1), math.hypot(a2, b2)
    h1 = math.degrees(math.atan2(b1, a1)) % 360
    h2 = math.degrees(math.atan2(b2, a2)) % 360
    dl, dc = L2 - L1, c2p - c1p
    dh = 0.0
    if c1p * c2p:
        dh = h2 - h1
        dh = dh - 360 if dh > 180 else dh + 360 if dh < -180 else dh
    dh = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    mean_l, mean_cp = (L1 + L2) / 2, (c1p + c2p) / 2
    mean_h = h1 + h2
    if c1p * c2p:
        mean_h = (h1 + h2) / 2 + (180 if abs(h1 - h2) > 180 and h1 + h2 < 360 else
                                  -180 if abs(h1 - h2) > 180 else 0)
    t = (1 - 0.17 * math.cos(math.radians(mean_h - 30)) + 0.24 * math.cos(math.radians(2 * mean_h))
         + 0.32 * math.cos(math.radians(3 * mean_h + 6)) - 0.20 * math.cos(math.radians(4 * mean_h - 63)))
    theta = 30 * math.exp(-(((mean_h - 275) / 25) ** 2))
    rc = 2 * math.sqrt(mean_cp ** 7 / (mean_cp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (mean_l - 50) ** 2 / math.sqrt(20 + (mean_l - 50) ** 2)
    sc, sh = 1 + 0.045 * mean_cp, 1 + 0.015 * mean_cp * t
    rt = -math.sin(math.radians(2 * theta)) * rc
    return math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh / sh) ** 2 + rt * (dc / sc) * (dh / sh))


# ---------- a theme's seed, checked ----------
def _number(value, bounds, label: str, whole: bool = False):
    low, high, _ = bounds
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ThemeError(f"{label} must be between {low:g} and {high:g}.")
    return int(round(value)) if whole else round(float(value), 2)


def tone(background) -> str | None:
    """'dark' or 'light' for a background (#RRGGBB or a tuple); None for one in between, which no glass suits."""
    lum = _lum(_rgb(background) if isinstance(background, str) else background)
    return "dark" if lum <= MAX_BG_LUMINANCE else "light" if lum >= MIN_LIGHT_LUMINANCE else None


def clean(data) -> dict:
    """A theme as the page or a file gives it, checked; only its known fields are kept."""
    if not isinstance(data, dict):
        raise ThemeError("A theme is a JSON object with a name and colours.")
    name = data.get("name")
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip() if isinstance(name, str) else ""
    if not 1 <= len(name) <= NAME_MAX:
        raise ThemeError(f"Give the theme a name of 1 to {NAME_MAX} characters.")
    seed = {"name": name}
    for key, label in (("background", "The background"), ("accent", "The accent")):
        value = data.get(key)
        if not (isinstance(value, str) and HEX_RE.fullmatch(value)):
            raise ThemeError(f"{label} must be a colour written like #1A2B3C.")
        seed[key] = value.upper()
    for key, label in ROLES:                       # a palette's own roles; each left out is worked out
        value = data.get(key)
        if value is None:
            continue
        if not (isinstance(value, str) and HEX_RE.fullmatch(value)):
            raise ThemeError(f"{label} must be a colour written like #1A2B3C, or left out to be worked out.")
        seed[key] = value.upper()
    glows = data.get("glows")
    if glows is not None:
        if not (isinstance(glows, list) and len(glows) == 3
                and all(isinstance(g, str) and HEX_RE.fullmatch(g) for g in glows)):
            raise ThemeError("The glows must be three colours written like #1A2B3C, or left out to match the accent.")
        seed["glows"] = [g.upper() for g in glows]
    seed["blur"] = _number(data.get("blur", BLUR[2]), BLUR, "The blur", whole=True)
    seed["tint"] = _number(data.get("tint", TINT[2]), TINT, "The glass tint")
    if tone(seed["background"]) is None:
        raise ThemeError("The background is in between dark and light: pick one darker than about #303030, or "
                         "lighter than about #C4C4C4 for a light theme.")
    return seed


# ---------- every colour the page uses, from the seed ----------
def _work(seed: dict) -> dict:
    """derive()'s result, with the colours hifz() and swatches() build on (as tuples)."""
    bg, accent = _rgb(seed["background"]), _rgb(seed["accent"])
    light = tone(bg) == "light"
    glows = [_rgb(g) for g in seed["glows"]] if seed.get("glows") else _glows_for(accent)
    blur, tint = seed.get("blur", BLUR[2]), seed.get("tint", TINT[2])
    hue, _, sat = _hls(bg)
    role = {key: _rgb(seed[key]) if seed.get(key) else None for key, _ in ROLES}
    panel, raised, border = role["panel"], role["raised"], role["border"]
    extended = light or any(role.values())          # a theme with roles, or a light one, gets the further checks
    move = {"darken": light}                        # where a colour goes to read: lighter, or darker on a light theme

    # What text sits on, at its brightest (its darkest, on a light theme): the page where the glows are strongest, the
    # glass over that (at its top, where the sheen is, or its foot on a light theme), and a pill on the page; and with
    # a raised panel, that panel and the glass over it (menus, dialogs, the Dashboard's dropdown).
    first = _over(glows[0], GLOW_ALPHA[0], bg)
    pages = (bg, first, _over(glows[1], GLOW_ALPHA[1], bg), _over(glows[2], GLOW_ALPHA[2], first))
    page = (min if light else max)(pages, key=_lum)
    tint_rgb = panel or _mix(bg, WHITE, 0.012)
    elev = raised or _mix(bg, WHITE, 0.04)
    edge = (role["text"] or _from_hls(hue, 0.16, min(0.25, sat))) if light else WHITE    # the tint of edges and fills
    # What pills and the faint fills of rows and buttons are tinted with: white on Glass (dark ink on a light theme),
    # the palette's own border on a theme with roles, whose muted text has no room for a white film over the glass
    fill = border if extended and border else edge
    sheen = LIGHT_SHEEN[1] if light else 0.07
    glass = _over(WHITE, sheen, _over(tint_rgb, tint, page))
    pill = _over(fill, PILL_ALPHA, page)
    surfaces = (glass, pill)
    if raised:
        surfaces += (elev, _over(WHITE, sheen, _over(tint_rgb, tint, elev)))
    if extended:                    # a pill or a ghost button on the glass, and on a menu or dialog
        surfaces += (_over(fill, PILL_ALPHA, glass), _over(fill, PILL_ALPHA, elev))

    # Text: near white in the background's own hue (a neutral background gets neutral text), lifted if needed; near
    # black on a light theme. A palette's own text is kept as it is, and checked.
    text = role["text"] or _readable(_from_hls(hue, 0.10 if light else 0.95, min(0.45, sat * 0.8)), surfaces, 7.0,
                                     **move)
    if role["text"] and role["muted"]:              # between the two: a dimmer shade of the palette's text, never
        text2 = _readable(_mix(text, tint_rgb, 0.18), surfaces, **move)          # a mix with another hue
    else:
        text2 = _readable(_from_hls(hue, 0.24 if light else 0.87, min(0.30, sat * 0.6)), surfaces, **move)
    label = role["muted"] or _readable(_from_hls(hue, 0.36 if light else 0.62, min(0.20, sat * 0.35)), surfaces,
                                       **move)

    # On the accent: a near-black in the background's hue, or white, whichever reads better (or the palette's own)
    dark_ink = _from_hls(hue, 0.04, min(0.6, sat))
    ink = role["on_accent"]
    if not ink:
        ink = dark_ink if contrast(dark_ink, accent) >= contrast(WHITE, accent) else WHITE
        if contrast(ink, accent) < MIN_TEXT:          # a mid-tone accent: pure black or white, whichever reads
            ink = max((BLACK, WHITE), key=lambda c: contrast(c, accent))     # better (one always reaches 4.58:1)
    a_hue, a_light, a_sat = _hls(accent)
    lighter_ink = _lum(ink) > _lum(accent) if role["on_accent"] else ink == WHITE
    hover = _from_hls(a_hue, a_light + (-0.06 if lighter_ink else 0.07), a_sat)
    if contrast(ink, hover) < MIN_TEXT:
        hover = accent

    accent_text = _readable(accent, surfaces, **move)                     # the accent as text (links, labels, focus)
    if light:                                                              # the accent, pale (deep, on a light theme):
        soft = _readable(_from_hls(a_hue, min(a_light, 0.34), a_sat), surfaces, darken=True)   # names, highlights
    else:
        soft = _readable(_from_hls(a_hue, max(a_light, 0.84), a_sat), surfaces)
    second = _shift(accent, 30)                                            # the "cyan" role: times, ready, thinking
    s_hue, _, s_sat = _hls(second)
    second_soft = _readable(_from_hls(s_hue, 0.30 if light else 0.84, s_sat), surfaces, **move)
    third = _shift(accent, -20)                                            # the "blue" role
    for bubble_alpha in (0.22, 0.18, 0.14, 0.10):                          # your bubble: as tinted as text allows
        bubble = _over(accent, bubble_alpha, page)
        if contrast(text, bubble) >= MIN_TEXT + 0.1:
            break
    orb2, orb4, orb5 = _mix(accent, WHITE, 0.75), _mix(accent, BLACK, 0.55), _mix(accent, BLACK, 0.78)
    shade = _mix(accent, BLACK, 0.88)
    sheen_css = (f"linear-gradient(180deg, rgba(255, 255, 255, {LIGHT_SHEEN[0]:g}), rgba(255, 255, 255, "
                 f"{LIGHT_SHEEN[1]:g}))" if light else
                 "linear-gradient(180deg, rgba(255, 255, 255, .07), rgba(255, 255, 255, .02))")

    tokens = {
        "--bg": _hex(bg),
        "--glow-1": _rgba(glows[0], GLOW_ALPHA[0]),
        "--glow-2": _rgba(glows[1], GLOW_ALPHA[1]),
        "--glow-3": _rgba(glows[2], GLOW_ALPHA[2]),
        "--accent": _hex(accent),
        "--accent-hover": _hex(hover),
        "--accent-ink": _hex(ink),
        "--accent-soft": _rgba(accent, 0.14),
        "--accent-text": _hex(accent_text),
        "--user-bubble": _rgba(accent, bubble_alpha),
        "--user-border": _rgba(accent, 0.5),
        "--text": _hex(text),
        "--text-2": _hex(text2),
        "--label": _hex(label),
        "--orb-1": "#FFFFFF",
        "--orb-2": _hex(orb2),
        "--orb-3": _hex(accent),
        "--orb-4": _hex(orb4),
        "--orb-5": _hex(orb5),
        "--orb-2-rgb": _triple(orb2),
        "--orb-shade-rgb": _triple(shade),
        "--glass-blur": f"{blur}px",
        "--glass-bg": sheen_css + ", " + _rgba(tint_rgb, tint),
        "--glass-fallback": _rgba(_mix(tint_rgb if panel else bg, WHITE, 0.06), 0.94),
        "--pill-border-hover": _rgba(accent, 0.45),
        "--scrim": _rgba(_mix(edge, bg, 0.3), 0.45) if light else _rgba(_mix(bg, BLACK, 0.6), 0.8),
        # the names the shared rules use
        "--bg-elev": _hex(elev),
        "--glass": _rgba(tint_rgb, tint),
        "--glass-strong": _rgba(tint_rgb, 0.82),
        "--violet": _hex(accent_text),
        "--violet-soft": _hex(soft),
        "--violet-rgb": _triple(accent),
        "--blue": _hex(third),
        "--blue-rgb": _triple(third),
        "--cyan": _hex(second),
        "--cyan-soft": _hex(second_soft),
        "--cyan-rgb": _triple(second),
        "--mint": _hex(OK_GREEN),
        "--mint-rgb": _triple(OK_GREEN),
        "--grad": f"linear-gradient(135deg, {_hex(accent)}, {_hex(accent)})",
        "--grad-soft": f"linear-gradient(135deg, {_rgba(accent, 0.16)}, {_rgba(second, 0.08)})",
        "--ring-focus": f"0 0 0 3px {_rgba(accent_text, 0.3)}",
    }
    toggle = _over(fill, TOGGLE_ALPHA, glass) if extended else glass    # the selected language button
    checks = [
        ("Text on glass", min(contrast(text, glass), contrast(text, toggle)), MIN_TEXT),
        ("Secondary text", min(contrast(text2, s) for s in surfaces), MIN_TEXT),
        ("Labels", min(contrast(label, s) for s in surfaces), MIN_TEXT),
        ("Text in your bubble", contrast(text, bubble), MIN_TEXT),
        ("Button text", contrast(ink, accent), MIN_TEXT),
        ("Accent text", min(contrast(accent_text, s) for s in surfaces), MIN_TEXT),
        ("Highlights", min(contrast(c, s) for c in (soft, second_soft) for s in surfaces), MIN_TEXT),
        ("Buttons stand out", contrast(accent, glass), MIN_BUTTONS),
    ]
    if extended:
        # The lines in the palette's border colour, or in the ink on a light theme without one
        lines, weight = (border, ROLE_LINES) if border else (edge, {k: a * 1.6 for k, a in EDGE_LINES.items()})
        tokens.update({k: _rgba(lines, round(a, 3)) for k, a in weight.items()})
        if raised:
            checks.insert(1, ("Text on menus and dialogs", min(contrast(text, s) for s in surfaces[2:]), MIN_TEXT))
        # The warning, error and "up" colours, as text on this theme's own surfaces (and the pastels the page writes
        # its pills, notes and the timer in on Glass, as readable here)
        fixed = {k: _readable(c, surfaces, **move) for k, c in {**STATUS_COLOURS, **STATUS_TEXTS}.items()}
        tokens.update({k: _hex(c) for k, c in fixed.items()})
        tokens["--mint-rgb"] = _triple(fixed["--mint"])
        checks.append(("Warnings and errors", min(contrast(c, s) for c in fixed.values() for s in surfaces), MIN_TEXT))
        # Faded items (a source's engines, a switched-off provider, a settled card, an interrupted reply) fade only as
        # far as the muted text still reads at 4.5:1: Glass fades them to 0.6-0.8 because its labels are at 6:1
        fade = next((f for f in FADES if min(contrast(_over(label, f, s), s) for s in surfaces) >= MIN_TEXT), 1.0)
        tokens.update({"--fade-floor": f"{fade:g}", "--ink-rgb": _triple(fill), "--pill-bg": _rgba(fill, PILL_ALPHA),
                       "--pill-bg-hover": _rgba(fill, 0.10),
                       "--thumb": _rgba(edge, 0.25) if light else "rgba(255, 255, 255, 0.08)"})
    if light:
        # The ink the shared rules tint their fills and edges with (white on the dark themes), the fields' fill, the
        # header's Hifz mark in a gold that shows on light glass, and shadows in the ink rather than black
        mark = _readable((233, 196, 106), surfaces, MIN_BUTTONS + 0.2, darken=True)
        dot = _readable((255, 209, 102), surfaces, MIN_BUTTONS + 0.2, darken=True)
        tokens.update({
            "--field-rgb": "255, 255, 255",
            "--alert-dot": _hex(dot),
            "--glass-shadow": f"inset 0 1px 0 rgba(255, 255, 255, .6), 0 12px 32px -14px {_rgba(edge, 0.28)}",
            "--shadow-sm": f"0 4px 14px {_rgba(edge, 0.12)}",
            "--shadow": f"0 14px 40px {_rgba(edge, 0.18)}",
            "--drop": _rgba(edge, 0.22),
            "--hifz-mark": _hex(mark),
        })
    return {"tokens": tokens, "tone": "light" if light else "dark", "roles": extended,
            "checks": [{"what": what, "ratio": round(ratio, 2), "min": low, "ok": ratio >= low}
                       for what, ratio, low in checks],
            "colours": {"bg": bg, "page": page, "panel": tint_rgb, "elev": elev, "text": text, "text2": text2,
                        "label": label, "accent": accent, "ink": ink, "border": border, "glows": glows,
                        "light": light, "explicit": bool(panel or raised)}}


def derive(seed: dict) -> dict:
    """The CSS variables for a cleaned seed, its tone ('dark' or 'light'), whether it's a palette (roles) or light
    theme, which the page marks with data-roles, and its contrast checks ({what, ratio, min, ok})."""
    worked = _work(seed)
    return {"tokens": worked["tokens"], "tone": worked["tone"], "roles": worked["roles"], "checks": worked["checks"]}


def swatches(seed: dict, colours: dict) -> list:
    """The theme as a strip of colours for the picker, from the background to the accent: a palette's own roles, or
    the glass, the glows as they show over the background, the muted text and the text."""
    c = colours
    if any(seed.get(key) for key, _ in ROLES):
        strip = [c["bg"], c["panel"], c["elev"], c["border"], c["label"], c["text"], c["accent"]]
    else:
        strip = [c["bg"], c["elev"], _over(c["glows"][0], 0.55, c["bg"]), _over(c["glows"][1], 0.55, c["bg"]),
                 c["label"], c["text"], c["accent"]]
    out = []
    for colour in strip:
        if colour is not None and _hex(colour) not in out:
            out.append(_hex(colour))
    return out


# ---------- the Hifz page's colours (index.html's .hz-scope variables) ----------
# Hifz Classic is the page's own look: these are the values the CSS gives .hz-scope, which a test keeps in step. The
# ones after the statuses name what its rules wrote as fixed colours before the page could take other themes.
HZ_PATTERN = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56' "
              "viewBox='0 0 56 56'%3E%3Cg fill='none' stroke='%23{colour}' stroke-width='0.8' stroke-opacity='{alpha}'"
              "%3E%3Cpath d='M28 11L33 16L40 16L40 23L45 28L40 33L40 40L33 40L28 45L23 40L16 40L16 33L11 28L16 23L16 "
              "16L23 16Z'/%3E%3Cpath d='M0 -17L5 -12L12 -12L12 -5L17 0L12 5L12 12L5 12L0 17L-5 12L-12 12L-12 5L-17 0L"
              "-12 -5L-12 -12L-5 -12Z'/%3E%3Cpath d='M0 39L5 44L12 44L12 51L17 56L12 61L12 68L5 68L0 73L-5 68L-12 68L"
              "-12 61L-17 56L-12 51L-12 44L-5 44Z'/%3E%3Cpath d='M56 -17L61 -12L68 -12L68 -5L73 0L68 5L68 12L61 12L56 "
              "17L51 12L44 12L44 5L39 0L44 -5L44 -12L51 -12Z'/%3E%3Cpath d='M56 39L61 44L68 44L68 51L73 56L68 61L68 "
              "68L61 68L56 73L51 68L44 68L44 61L39 56L44 51L44 44L51 44Z'/%3E%3Cpath d='M28 20.4L30.2 22.6L33.4 22.6L"
              "33.4 25.8L35.6 28L33.4 30.2L33.4 33.4L30.2 33.4L28 35.6L25.8 33.4L22.6 33.4L22.6 30.2L20.4 28L22.6 "
              "25.8L22.6 22.6L25.8 22.6Z'/%3E%3Cpath d='M28 0V11M28 45V56M0 28H11M45 28H56'/%3E%3C/g%3E%3C/svg%3E\")")
HZ_STAR = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cpath "
           "fill='%23{colour}' d='M50 2L64.1 16.1L83.9 16.1L83.9 35.9L98 50L83.9 64.1L83.9 83.9L64.1 83.9L50 98L35.9 "
           "83.9L16.1 83.9L16.1 64.1L2 50L16.1 35.9L16.1 16.1L35.9 16.1Z'/%3E%3C/svg%3E\")")   # behind today's portion
HIFZ_CLASSIC = {
    "--hz-card": "rgba(8, 48, 37, 0.8)", "--hz-card-2": "rgba(13, 64, 50, 0.88)",
    "--hz-line": "rgba(233, 196, 106, 0.22)", "--hz-line-2": "rgba(233, 196, 106, 0.48)",
    "--hz-gold": "#E9C46A", "--hz-ink": "#1E1606", "--hz-ivory": "#F6EED8", "--hz-text-2": "#D3E3D8",
    "--hz-muted": "#A3BDB0",
    "--hz-fresh": "#E9C46A", "--hz-steady": "#B9A35A", "--hz-soon": "#7FAE95", "--hz-over": "#E08A5E",
    "--hz-weak": "#FF7A6B",
    "--hz-pattern": HZ_PATTERN.format(colour="E9C46A", alpha="0.13"),
    "--hz-today-star": HZ_STAR.format(colour="E9C46A"),
    "--hz-bg": "#04211A", "--hz-gold-rgb": "233, 196, 106", "--hz-shade-rgb": "2, 19, 15",
    "--hz-deep-rgb": "4, 33, 26", "--hz-drop-rgb": "0, 0, 0", "--hz-ivory-rgb": "246, 238, 216",
    "--hz-star-ink": "#1E1606", "--hz-star-ink-rgb": "30, 22, 6", "--hz-weak-rgb": "255, 122, 107",
    "--hz-weak-text": "#FFB2A8", "--hz-over-text": "#E18D62", "--hz-done": "#B9C7BB", "--hz-btn-1": "#F1D27F",
    "--hz-btn-2": "#D6A84C", "--hz-view-1": "#062B21", "--hz-view-2": "#031914", "--hz-view-3": "#02130F",
    "--hz-dash-1": "#0B3F31", "--hz-bar-1": "#0C4334", "--hz-bar-2": "#052A20", "--hz-ring-1": "#0E4A39",
    "--hz-ring-2": "#05281F", "--hz-drawer": "#03201A", "--hz-hl-edge": "rgba(233, 196, 106, 0.5)",
}
HZ_STATUSES = ("fresh", "steady", "soon", "over", "weak")
HZ_GLOW_AT_CONTENT = 0.08          # the view's gold glow (0.16 at its peak, above the top edge) where text begins
HZ_HIGHLIGHT = 0.17                # the ayah being recited: the accent at its strongest across the highlight
HZ_TINT, HZ_SOFT_TINT = 0.20, 0.14 # the accent's tint under text: a pill, a pressed toggle (0.20); a chosen row (0.14)
HZ_DONE_TINT = 0.12                # Dua & Zikr: a dua said its full count, its card lit faintly in the accent
HZ_COUNTER = 0.55                  # Dua & Zikr: the counter's face, the view's deep shade over the card
MIN_STATUS = 3.0                   # a status colour against what it sits on (WCAG's 3:1 for graphics)
MIN_APART = 10.0                   # CIEDE2000 between any two of the six (Classic's closest, fresh and steady: 10.5)
MIN_WEAK_APART = 11.0              # weak against every other (Classic: 11.9)


def _colour(value: str) -> tuple:
    """'#RRGGBB' or 'rgba(r, g, b, a)' → ((r, g, b), alpha)."""
    if value.startswith("#"):
        return _rgb(value), 1.0
    numbers = [float(x) for x in value[value.index("(") + 1:value.index(")")].split(",")]
    return tuple(numbers[:3]), numbers[3] if len(numbers) > 3 else 1.0


def _numbers(value: str) -> tuple:
    return tuple(float(x) for x in value.split(","))


def _hz_surfaces(v: dict) -> dict:
    """What the Hifz page's text and status colours sit on, from its variables: the view (its top under the glow, its
    middle, its foot), the cards over it (both ends of their gradient), the highlighted ayah, the player, the
    Dashboard's card, the reminder, the drawer (the surah list on a narrow window, and the theme list), and the
    accent's tints under text (pills, pressed toggles, chosen rows)."""
    gold = _numbers(v["--hz-gold-rgb"])
    view1, view2, view3 = (_rgb(v[f"--hz-view-{i}"]) for i in (1, 2, 3))
    top = _over(gold, HZ_GLOW_AT_CONTENT, view1)
    card1, card2 = _colour(v["--hz-card"]), _colour(v["--hz-card-2"])
    views = [top, view2, view3]
    cards = [_over(*card2, top), _over(*card1, view2), _over(*card2, view2), _over(*card1, top)]
    dash = [_rgb(v["--hz-dash-1"]), _rgb(v["--hz-bg"])]
    return {"gold": gold, "views": views, "cards": cards, "dash": dash,
            "highlight": [_over(gold, HZ_HIGHLIGHT, top), _over(gold, HZ_HIGHLIGHT, view2)],
            "bars": [_rgb(v["--hz-bar-1"]), _rgb(v["--hz-bar-2"])],
            "ring": [_rgb(v["--hz-ring-1"]), _rgb(v["--hz-ring-2"])], "drawer": [_rgb(v["--hz-drawer"])],
            "tints": [_over(gold, HZ_TINT, x) for x in cards + views + dash],     # a pill, a pressed toggle
            # a chosen row: the surah list and the reciters sit on the shaded side panels (or the drawer)
            "soft_tints": [_over(gold, HZ_SOFT_TINT, x) for x in
                           [_over(_numbers(v["--hz-shade-rgb"]), 0.5, y) for y in views] + [_rgb(v["--hz-drawer"])]]}


def hifz_checks(v: dict) -> list:
    """The Hifz page's contrast checks for a full set of its variables ({what, ratio, min, ok}): the Arabic text and
    the translation everywhere, the highlighted ayah, the buttons, each status colour against its card with its juz
    number on it, and the six statuses apart from each other, weak furthest."""
    s = _hz_surfaces(v)
    c = lambda name: _rgb(v[name])
    worst = lambda colour, where: min(contrast(colour, x) for x in where)
    text_on = s["views"] + s["cards"] + s["bars"] + s["dash"] + s["ring"] + s["drawer"]
    edge, edge_alpha = _colour(v["--hz-hl-edge"])
    tinted = min(worst(c(k), s["tints"]) for k in ("--hz-gold", "--hz-ivory", "--hz-text-2"))
    out = [
        ("Arabic text", worst(c("--hz-ivory"), text_on + s["highlight"]), MIN_TEXT),
        ("Translation text", worst(c("--hz-text-2"), text_on + s["highlight"]), MIN_TEXT),
        ("Muted text", worst(c("--hz-muted"), text_on), MIN_TEXT),
        ("Accent as text", worst(c("--hz-gold"), text_on), MIN_TEXT),
        ("Highlighted ayah: its Arabic", worst(c("--hz-ivory"), s["highlight"]), MIN_TEXT),
        ("Highlighted ayah: the word being recited", worst(c("--hz-gold"), s["highlight"]), MIN_TEXT),
        ("Highlighted ayah: its edge", min(contrast(_over(edge, edge_alpha, x), x) for x in s["views"]), MIN_STATUS),
        ("Ayahs already recited", worst(c("--hz-done"), s["views"] + s["highlight"]), MIN_TEXT),
        ("Button text", worst(c("--hz-ink"), [c("--hz-btn-1"), c("--hz-btn-2"), c("--hz-gold")]), MIN_TEXT),
        ("Text on the accent's tints (pills, pressed buttons)", tinted, MIN_TEXT),
        ("Muted text on a chosen row", worst(c("--hz-muted"), s["soft_tints"]), MIN_TEXT),
    ]
    done = [_over(s["gold"], HZ_DONE_TINT, x) for x in s["cards"]]
    face = [_over(_numbers(v["--hz-deep-rgb"]), HZ_COUNTER, x) for x in s["cards"] + done]
    out += [("Dua & Zikr: the Arabic of a finished dua", worst(c("--hz-ivory"), done), MIN_TEXT),
            ("Dua & Zikr: its meaning on a finished dua", worst(c("--hz-text-2"), done), MIN_TEXT),
            ("Dua & Zikr: the counter's number", worst(c("--hz-ivory"), face), MIN_TEXT),
            ("Dua & Zikr: the counter's count to go", worst(c("--hz-muted"), face), MIN_TEXT)]
    under = s["cards"] + s["dash"]
    for key, word in (("fresh", "Fresh"), ("steady", "Steady"), ("soon", "Due soon"), ("over", "Past its cycle")):
        out.append((f"{word}: against the card", worst(c(f"--hz-{key}"), under), MIN_STATUS))
        out.append((f"{word}: the juz number on it", contrast(c("--hz-star-ink"), c(f"--hz-{key}")), MIN_TEXT))
    out.append(("Past its cycle, as text", worst(c("--hz-over-text"), s["cards"]), MIN_TEXT))
    never = _over(s["gold"], 0.07, s["cards"][0])          # a juz not revised yet: the faint fill .hz-star gives it
    out.append(("Not yet: the juz number", contrast(c("--hz-ivory"), never), MIN_TEXT))
    weak = c("--hz-weak")
    out.append(("Weak: against the card", worst(weak, under), MIN_STATUS))
    out.append(("Weak: against its ring", contrast(weak, c("--hz-bg")), MIN_STATUS))
    pill = _over(_numbers(v["--hz-weak-rgb"]), 0.16, s["cards"][0])
    out.append(("Weak, as text", worst(c("--hz-weak-text"), [pill] + s["cards"]), MIN_TEXT))
    six = {**{k: c(f"--hz-{k}") for k in HZ_STATUSES}, "not yet": never}
    apart = {f"{a} and {b}": delta_e(six[a], six[b]) for a, b in itertools.combinations(six, 2)}
    closest = min(apart, key=apart.get)
    out.append((f"Statuses apart (closest: {closest})", apart[closest], MIN_APART))
    weakest = min((p for p in apart if "weak" in p.split(" and ")), key=apart.get)
    out.append((f"Weak stands out (closest: {weakest})", apart[weakest], MIN_WEAK_APART))
    return [{"what": what, "ratio": round(ratio, 2), "min": low, "ok": ratio >= low} for what, ratio, low in out]


def _status_fills(under, ring, star_ink, light: bool) -> dict:
    """Classic's five status colours moved together in lightness (lighter on a dark theme, darker on a light one) by
    the least that lets each stand 3:1 off what it sits on with its juz number readable on it, keeping them as far
    apart as Classic keeps them. Classic's own colours when they already fit."""
    base = {k: _rgb(HIFZ_CLASSIC[f"--hz-{k}"]) for k in HZ_STATUSES}

    def fits(fills):
        for key, colour in fills.items():
            if min(contrast(colour, x) for x in under) < MIN_STATUS + 0.05:
                return False
            if key == "weak" and contrast(colour, ring) < MIN_STATUS + 0.05:
                return False
            if key != "weak" and contrast(star_ink, colour) < MIN_TEXT + 0.05:
                return False
        return all(delta_e(fills[a], fills[b]) >= (MIN_WEAK_APART if "weak" in (a, b) else MIN_APART) + 0.2
                   for a, b in itertools.combinations(fills, 2))
    for step in range(121):
        dl = (-0.5 if light else 0.5) * step
        fills = {k: _at(colour, dl) if dl else colour for k, colour in base.items()}
        if fits(fills):
            return fills
    return base                                     # no shift fits: the checks will say so


def _hifz(colours: dict) -> dict:
    """The Hifz page's variables for a theme: its colours on Classic's structure (a view lit from the top, cards a
    step lighter, lines and the star pattern in the accent), every text colour moved only as far as it must to read,
    and the status colours from _status_fills."""
    c = colours
    light = c["light"]
    bg, panel, elev, accent = c["bg"], c["panel"], c["elev"], c["accent"]
    if light:
        view1, view2, view3 = _mix(bg, WHITE, 0.5), bg, _mix(bg, panel, 0.35)
        card1, card2 = _mix(panel, WHITE, 0.45), _mix(elev, WHITE, 0.55)
        dash1, bar1, bar2 = _mix(panel, WHITE, 0.3), _mix(panel, WHITE, 0.55), _mix(panel, WHITE, 0.25)
        ring1, ring2 = _mix(elev, WHITE, 0.5), _mix(panel, WHITE, 0.2)
        shade, deep, drop, drawer = _mix(bg, WHITE, 0.6), WHITE, _mix(c["text"], bg, 0.35), _mix(bg, WHITE, 0.4)
    else:
        if not c["explicit"]:      # a theme with no panel colours: cards lifted off the background, as Classic's are
            panel, elev = _at(_mix(bg, accent, 0.05), 9), _at(_mix(bg, accent, 0.08), 17)
        view1, view2, view3 = _mix(bg, panel, 0.55), bg, _mix(bg, BLACK, 0.3)
        card1, card2 = panel, _mix(panel, elev, 0.45)
        dash1, bar1, bar2 = _mix(panel, elev, 0.6), _mix(panel, elev, 0.55), _mix(bg, panel, 0.6)
        ring1, ring2 = _mix(panel, elev, 0.7), _mix(bg, panel, 0.7)
        shade, deep, drop, drawer = _mix(bg, BLACK, 0.35), _mix(bg, panel, 0.6), BLACK, _mix(bg, panel, 0.4)
    v = {"--hz-bg": _hex(bg), "--hz-card": _rgba(card1, 0.8), "--hz-card-2": _rgba(card2, 0.88),
         "--hz-view-1": _hex(view1), "--hz-view-2": _hex(view2), "--hz-view-3": _hex(view3), "--hz-dash-1": _hex(dash1),
         "--hz-bar-1": _hex(bar1), "--hz-bar-2": _hex(bar2), "--hz-ring-1": _hex(ring1), "--hz-ring-2": _hex(ring2),
         "--hz-drawer": _hex(drawer), "--hz-gold-rgb": _triple(accent), "--hz-shade-rgb": _triple(shade),
         "--hz-deep-rgb": _triple(deep), "--hz-drop-rgb": _triple(drop),
         "--hz-line": _rgba(accent, 0.30 if light else 0.22), "--hz-line-2": _rgba(accent, 0.6 if light else 0.48),
         "--hz-pattern": HZ_PATTERN.format(colour=_hex(accent)[1:], alpha="0.22" if light else "0.13"),
         "--hz-today-star": HZ_STAR.format(colour=_hex(accent)[1:])}
    s = _hz_surfaces({**HIFZ_CLASSIC, **v})
    everywhere = s["views"] + s["cards"] + s["bars"] + s["dash"] + s["ring"] + s["drawer"]
    readable = lambda colour, where, low=MIN_TEXT + 0.05: _readable(colour, where, low, darken=light)
    tinted = s["highlight"] + s["tints"]                          # the accent's tints: the ayah recited, pills, toggles
    gold = readable(accent, everywhere + tinted)                  # the accent as text: names, labels, the word recited
    ivory = readable(c["text"], everywhere + tinted)
    # The buttons: the accent itself, a shade either side across the gradient, as far as its text still reads on it
    ink = c["ink"]
    if contrast(ink, accent) < MIN_TEXT + 0.05:
        ink = max((BLACK, WHITE), key=lambda x: contrast(x, accent))
    a_hue, a_light, a_sat = _hls(accent)

    def stop(shift):
        for k in range(10, -1, -1):
            colour = _from_hls(a_hue, a_light + shift * k / 10, a_sat)
            if contrast(ink, colour) >= MIN_TEXT + 0.05:
                return colour
        return accent
    v.update({"--hz-gold": _hex(gold), "--hz-ink": _hex(ink), "--hz-ivory": _hex(ivory),
              "--hz-ivory-rgb": _triple(ivory),
              "--hz-text-2": _hex(readable(c["text2"], everywhere + tinted)),
              "--hz-muted": _hex(readable(c["label"], everywhere + s["soft_tints"])),
              "--hz-done": _hex(readable(_mix(c["text"], c["label"], 0.6), s["views"] + s["highlight"])),
              "--hz-btn-1": _hex(stop(0.05)), "--hz-btn-2": _hex(stop(-0.05))})
    star_ink = WHITE if light else _rgb(HIFZ_CLASSIC["--hz-star-ink"])
    fills = _status_fills(s["cards"] + [_rgb(v["--hz-dash-1"]), bg], bg, star_ink, light)
    v.update({f"--hz-{k}": _hex(colour) for k, colour in fills.items()})
    v.update({"--hz-star-ink": _hex(star_ink), "--hz-star-ink-rgb": _triple(star_ink),
              "--hz-weak-rgb": _triple(fills["weak"]),
              "--hz-weak-text": _hex(readable(_rgb(HIFZ_CLASSIC["--hz-weak-text"]),
                                              [_over(fills["weak"], 0.16, s["cards"][0])] + s["cards"])),
              "--hz-over-text": _hex(readable(fills["over"], s["cards"]))})
    for alpha in [a / 100 for a in range(50, 101, 2)]:     # the highlighted ayah's edge, strong enough to see
        if min(contrast(_over(gold, alpha, x), x) for x in s["views"]) >= MIN_STATUS + 0.05:
            break
    v["--hz-hl-edge"] = _rgba(gold, alpha)
    checks = hifz_checks({**HIFZ_CLASSIC, **v})
    return {"tokens": v, "checks": checks, "ok": all(x["ok"] for x in checks),
            "swatches": [_hex(view2), _hex(card2), _hex(gold), v["--hz-ivory"]]}


def hifz_classic() -> dict:
    """Hifz Classic, as the page's CSS has it, with its checks."""
    checks = hifz_checks(HIFZ_CLASSIC)
    return {"id": "classic", "name": "Hifz Classic", "tokens": {}, "checks": checks, "ok": all(x["ok"] for x in checks),
            "swatches": ["#031914", "#0D4032", "#E9C46A", "#F6EED8"]}


# ---------- the files ----------
def _path(theme_id: str) -> str:
    return os.path.join(THEMES_DIR, theme_id + ".json")


def _check_id(theme_id) -> None:
    if not (isinstance(theme_id, str) and ID_RE.fullmatch(theme_id)) or theme_id in RESERVED:
        raise ThemeError("There's no theme by that name.")


def _usable(theme_id: str, seed: dict) -> dict:
    """The theme as the page gets it; refused if a check fails (a hand-edited file can be too dim). Its Hifz colours
    come along, with their own checks: a theme whose Hifz colours fall short is still a theme, just not offered on
    the Hifz page."""
    worked = _work(seed)
    weak = [c for c in worked["checks"] if not c["ok"]]
    if weak:
        raise ThemeError("Not readable enough: " + ", ".join(
            f"{c['what'].lower()} {c['ratio']:g}:1 (needs {c['min']:g})" for c in weak) + ".")
    return {"id": theme_id, **seed, "tokens": worked["tokens"], "tone": worked["tone"], "roles": worked["roles"],
            "checks": worked["checks"],
            "swatches": swatches(seed, worked["colours"]), "preset": theme_id in PRESETS,
            "hifz": _hifz(worked["colours"])}


def _ids() -> list:
    if not os.path.isdir(THEMES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(THEMES_DIR) if f.endswith(".json"))


def listing() -> dict:
    """Every usable theme file, sorted by name, with its colours worked out; what's wrong with the rest (a broken
    file is reported, never silently skipped); and Hifz Classic's checks."""
    themes, problems = [], []
    for theme_id in _ids():
        name = theme_id + ".json"
        if not ID_RE.fullmatch(theme_id) or theme_id in RESERVED:
            problems.append(f"{name}: a theme's file name must be lower-case letters, digits and dashes, "
                            f"and not one of {', '.join(RESERVED)}")
            continue
        if len(themes) >= MAX_THEMES:
            problems.append(f"{name}: only {MAX_THEMES} themes are listed")
            continue
        try:
            if os.path.getsize(_path(theme_id)) > MAX_FILE_BYTES:
                raise ThemeError("too big for a theme file.")
            with open(_path(theme_id), encoding="utf-8") as f:
                themes.append(_usable(theme_id, clean(json.load(f))))
        except (OSError, ValueError) as e:         # ThemeError and a JSON syntax error are both ValueErrors
            problems.append(f"{name}: {e}")
    themes.sort(key=lambda t: t["name"].lower())
    return {"themes": themes, "problems": problems, "hifz_classic": hifz_classic()}


def _write(theme_id: str, seed: dict) -> dict:
    theme = _usable(theme_id, seed)                 # an unreadable theme is refused before anything is written
    os.makedirs(THEMES_DIR, exist_ok=True)
    body = json.dumps({k: seed[k] for k in FIELDS if k in seed}, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=THEMES_DIR, prefix=".theme-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, 0o644)
        os.replace(tmp, _path(theme_id))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return theme


def _free_id(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24].strip("-") or "theme"
    taken = set(_ids()) | set(RESERVED)
    if base not in taken:
        return base
    for n in range(2, 1000):
        if f"{base}-{n}" not in taken:
            return f"{base}-{n}"
    raise ThemeError("There are too many themes with that name.")


def create(data) -> dict:
    """A new theme file, named after the theme (ocean.json, ocean-2.json…)."""
    seed = clean(data)
    if len(_ids()) >= MAX_THEMES:
        raise ThemeError(f"There are already {MAX_THEMES} themes: delete one first.")
    return _write(_free_id(seed["name"]), seed)


def update(theme_id: str, data) -> dict:
    """An existing theme changed; its file name (id) stays when it's renamed."""
    _check_id(theme_id)
    if not os.path.exists(_path(theme_id)):
        raise KeyError(theme_id)
    return _write(theme_id, clean(data))


def delete(theme_id: str) -> bool:
    _check_id(theme_id)
    try:
        os.remove(_path(theme_id))
        return True
    except FileNotFoundError:
        return False
