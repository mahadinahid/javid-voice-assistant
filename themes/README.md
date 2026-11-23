# Colour themes

Each `.json` file here is a colour theme: Javid's Glass look in your own colours. They're listed under the header's
theme button, after the built-in looks (Orbit, Glass and Overdrive), each as a strip of its colours, and the list
re-reads this folder each time it opens. Javid ships fourteen: Ember, Graphite, Ocean and Orchid, and the palettes
Slate, Royal Blue, Plum, Mist (the light one), Teal Rust, Berry Coral, Violet, Navy Amber, Coffee and Olive.

Make one in the app (theme button → **+ New theme**, or ⧉ beside a theme to start from it: the page changes as you
pick, and you can edit or delete it later with ✎), or write a file yourself. The simplest:

```json
{
  "name": "Ocean",
  "background": "#06101C",
  "accent": "#5AB2FF",
  "glows": ["#1D4ED8", "#06B6D4", "#8B5CF6"],
  "blur": 18,
  "tint": 0.58
}
```

A palette can also name its own roles (any it leaves out are worked out):

```json
{
  "name": "Slate",
  "background": "#06141B",
  "panel": "#11212D",
  "raised": "#253745",
  "border": "#4A5C6A",
  "text": "#CCD0CF",
  "muted": "#9BA8AB",
  "accent": "#9BA8AB",
  "on_accent": "#06141B",
  "glows": ["#253745", "#11212D", "#4A5C6A"],
  "blur": 18,
  "tint": 0.72
}
```

- **name**: 1 to 24 characters, shown in the list. The file name is the theme's id: lower-case letters, digits and
  dashes (`ocean.json`), and not `orbit`, `glass`, `aurora`, `overdrive`, `classic` or `preview`.
- **background**: dark (about `#303030` or darker) or light (about `#C4C4C4` or lighter: a light theme, with dark
  text). A mid-grey in between is refused, since neither the dark glass's edges nor the light glass's show on it.
- **accent**: the buttons, the orb, your chat bubble and the highlights.
- **glows** (optional): the three soft lights behind everything (top left, bottom right, top). Leave them out and they
  follow the accent.
- **blur** (0 to 28, default 18) and **tint** (0.40 to 0.85, default 0.58): how frosted and how dark the glass is.
- **panel**, **raised**, **border**, **text**, **muted**, **on_accent** (optional, a palette's roles): the glass's
  colour, menus and dialogs, the lines, the text, the dimmer labels, and the text on the accent's buttons. They're
  used as given. A theme with roles tints its pills and faint fills with its own border colour, and fades nothing
  further than its muted text can take.

Every other colour (text and labels when not given, the text on buttons, the orb's shades) is worked out from these by
`themes.py`, which checks that text reads at 4.5:1 or better wherever it appears and that buttons stand out at 3:1. A
file that fails a check or isn't valid isn't used, and the list says which file and why.

**The Hifz page** has its own theme, chosen with the Theme button in its header: Hifz Classic (its green and gold), or
any of the fourteen presets here, whose colours `themes.py` maps onto the page's own roles (the background and its star
pattern, the cards, the Arabic text and the translation, the accent, the player). The juz status colours stay Hifz
Classic's set, moved in lightness only as far as contrast asks.
