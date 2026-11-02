"""Colour themes: palettes on the Glass look, one small JSON file each in themes/ (2026-10-26).

A theme file gives a name, a background, an accent, optionally three glow colours, and the glass's blur and tint.
derive() works out every other colour the page uses from those and checks that text stays readable: at least 4.5:1
wherever it appears, measured at the brightest glow and through the glass's tint, as Glass itself was checked. A
theme that can't be made readable is refused, so no theme can leave Javid unreadable. The page sets the derived
colours as CSS variables on <html>; the looks themselves (Glass, Overdrive) stay CSS in index.html.
Light backgrounds aren't supported: the glass's white edges and fills are made for dark ones.
"""

import colorsys
import json
import os
import re
import tempfile

THEMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")
RESERVED = ("glass", "aurora", "overdrive", "classic", "preview")   # the looks in index.html, the names older
                                                                     # versions saved, and the editor's route
ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")
NAME_MAX = 24
MAX_THEMES = 40
MAX_FILE_BYTES = 16_000
BLUR = (0, 28, 18)                 # px: lowest, highest, default (Glass's)
TINT = (0.40, 0.85, 0.58)          # the glass's tint opacity: lowest, highest, default (Glass's)
MAX_BG_LUMINANCE = 0.03            # about #303030: lighter, and the glass's white edges and fills stop working
MIN_TEXT = 4.5                     # WCAG AA for text
MIN_BUTTONS = 3.0                  # a button against the glass it sits on (WCAG's 3:1 for controls)
GLOW_ALPHA = (0.24, 0.16, 0.09)    # top left, bottom right, top centre: Glass's
FIELDS = ("name", "background", "accent", "glows", "blur", "tint")
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
OK_GREEN = (74, 222, 128)          # "up" and "serving": the same green in every colour theme


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


def _readable(rgb, surfaces, minimum: float = MIN_TEXT + 0.1) -> tuple:
    """`rgb`, lightened in its own hue until it reads at `minimum` on every surface (they're all dark)."""
    hue, light, sat = _hls(rgb)
    colour = rgb
    while light < 0.985 and min(contrast(colour, s) for s in surfaces) < minimum:
        light += 0.01
        colour = _from_hls(hue, light, sat)
    return colour


def _glows_for(accent) -> list:
    """Glows that match the accent, as Glass's do its own: a deeper shade, a neighbour on each side."""
    return [_mix(accent, BLACK, 0.2), _shift(accent, 30), _shift(accent, -45)]


# ---------- a theme's seed, checked ----------
def _number(value, bounds, label: str, whole: bool = False):
    low, high, _ = bounds
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ThemeError(f"{label} must be between {low:g} and {high:g}.")
    return int(round(value)) if whole else round(float(value), 2)


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
    glows = data.get("glows")
    if glows is not None:
        if not (isinstance(glows, list) and len(glows) == 3
                and all(isinstance(g, str) and HEX_RE.fullmatch(g) for g in glows)):
            raise ThemeError("The glows must be three colours written like #1A2B3C, or left out to match the accent.")
        seed["glows"] = [g.upper() for g in glows]
    seed["blur"] = _number(data.get("blur", BLUR[2]), BLUR, "The blur", whole=True)
    seed["tint"] = _number(data.get("tint", TINT[2]), TINT, "The glass tint")
    if _lum(_rgb(seed["background"])) > MAX_BG_LUMINANCE:
        raise ThemeError("The background is too light: pick a darker one (light themes aren't supported yet).")
    return seed


# ---------- every colour the page uses, from the seed ----------
def derive(seed: dict) -> dict:
    """The CSS variables for a cleaned seed, and its contrast checks ({what, ratio, min, ok})."""
    bg, accent = _rgb(seed["background"]), _rgb(seed["accent"])
    glows = [_rgb(g) for g in seed["glows"]] if seed.get("glows") else _glows_for(accent)
    blur, tint = seed.get("blur", BLUR[2]), seed.get("tint", TINT[2])
    hue, _, sat = _hls(bg)

    # What text sits on, at its brightest: the page where the glows are strongest, the glass over that (at its top,
    # where the sheen is), and a pill on the page.
    first = _over(glows[0], GLOW_ALPHA[0], bg)
    page = max((bg, first, _over(glows[1], GLOW_ALPHA[1], bg), _over(glows[2], GLOW_ALPHA[2], first)), key=_lum)
    tint_rgb = _mix(bg, WHITE, 0.012)
    glass = _over(WHITE, 0.07, _over(tint_rgb, tint, page))
    pill = _over(WHITE, 0.06, page)
    surfaces = (glass, pill)

    # Text: near white in the background's own hue (a neutral background gets neutral text), lifted if needed
    text = _readable(_from_hls(hue, 0.95, min(0.45, sat * 0.8)), surfaces, 7.0)
    text2 = _readable(_from_hls(hue, 0.87, min(0.30, sat * 0.6)), surfaces)
    label = _readable(_from_hls(hue, 0.62, min(0.20, sat * 0.35)), surfaces)

    # On the accent: a near-black in the background's hue, or white, whichever reads better
    dark_ink = _from_hls(hue, 0.04, min(0.6, sat))
    ink = dark_ink if contrast(dark_ink, accent) >= contrast(WHITE, accent) else WHITE
    if contrast(ink, accent) < MIN_TEXT:              # a mid-tone accent: pure black or white, whichever reads
        ink = max((BLACK, WHITE), key=lambda c: contrast(c, accent))     # better (one always reaches 4.58:1)
    a_hue, a_light, a_sat = _hls(accent)
    hover = _from_hls(a_hue, a_light + (-0.06 if ink == WHITE else 0.07), a_sat)
    if contrast(ink, hover) < MIN_TEXT:
        hover = accent

    accent_text = _readable(accent, surfaces)                              # the accent as text (links, labels, focus)
    soft = _readable(_from_hls(a_hue, max(a_light, 0.84), a_sat), surfaces)   # pale accent: names, highlights
    second = _shift(accent, 30)                                            # the "cyan" role: times, ready, thinking
    s_hue, _, s_sat = _hls(second)
    second_soft = _readable(_from_hls(s_hue, 0.84, s_sat), surfaces)
    third = _shift(accent, -20)                                            # the "blue" role
    for bubble_alpha in (0.22, 0.18, 0.14, 0.10):                          # your bubble: as tinted as text allows
        bubble = _over(accent, bubble_alpha, page)
        if contrast(text, bubble) >= MIN_TEXT + 0.1:
            break
    orb2, orb4, orb5 = _mix(accent, WHITE, 0.75), _mix(accent, BLACK, 0.55), _mix(accent, BLACK, 0.78)
    shade = _mix(accent, BLACK, 0.88)

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
        "--glass-bg": "linear-gradient(180deg, rgba(255, 255, 255, .07), rgba(255, 255, 255, .02)), "
                      + _rgba(tint_rgb, tint),
        "--glass-fallback": _rgba(_mix(bg, WHITE, 0.06), 0.94),
        "--pill-border-hover": _rgba(accent, 0.45),
        "--scrim": _rgba(_mix(bg, BLACK, 0.6), 0.8),
        # the names the shared rules use
        "--bg-elev": _hex(_mix(bg, WHITE, 0.04)),
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
    checks = [
        ("Text on glass", contrast(text, glass), MIN_TEXT),
        ("Secondary text", min(contrast(text2, s) for s in surfaces), MIN_TEXT),
        ("Labels", min(contrast(label, s) for s in surfaces), MIN_TEXT),
        ("Text in your bubble", contrast(text, bubble), MIN_TEXT),
        ("Button text", contrast(ink, accent), MIN_TEXT),
        ("Accent text", min(contrast(accent_text, s) for s in surfaces), MIN_TEXT),
        ("Highlights", min(contrast(c, s) for c in (soft, second_soft) for s in surfaces), MIN_TEXT),
        ("Buttons stand out", contrast(accent, glass), MIN_BUTTONS),
    ]
    return {"tokens": tokens,
            "checks": [{"what": what, "ratio": round(ratio, 2), "min": low, "ok": ratio >= low}
                       for what, ratio, low in checks]}


# ---------- the files ----------
def _path(theme_id: str) -> str:
    return os.path.join(THEMES_DIR, theme_id + ".json")


def _check_id(theme_id) -> None:
    if not (isinstance(theme_id, str) and ID_RE.fullmatch(theme_id)) or theme_id in RESERVED:
        raise ThemeError("There's no theme by that name.")


def _usable(theme_id: str, seed: dict) -> dict:
    """The theme as the page gets it; refused if a check fails (a hand-edited file can be too dim)."""
    derived = derive(seed)
    weak = [c for c in derived["checks"] if not c["ok"]]
    if weak:
        raise ThemeError("Not readable enough: " + ", ".join(
            f"{c['what'].lower()} {c['ratio']:g}:1 (needs {c['min']:g})" for c in weak) + ".")
    return {"id": theme_id, **seed, **derived}


def _ids() -> list:
    if not os.path.isdir(THEMES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(THEMES_DIR) if f.endswith(".json"))


def listing() -> dict:
    """Every usable theme file, sorted by name, with its colours worked out; and what's wrong with the rest (a
    broken file is reported, never silently skipped)."""
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
    return {"themes": themes, "problems": problems}


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
