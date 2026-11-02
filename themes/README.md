# Colour themes

Each `.json` file here is a colour theme: Javid's Glass look in your own colours. They're listed under the header's
theme button, after the built-in looks (Orbit, Glass and Overdrive), and the list re-reads this folder each time it
opens.

Make one in the app (theme button → **+ New theme**: the page changes as you pick, and you can edit or delete it later
with ✎), or write a file yourself:

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

- **name**: 1 to 24 characters, shown in the list. The file name is the theme's id: lower-case letters, digits and
  dashes (`ocean.json`), and not `orbit`, `glass`, `aurora`, `overdrive`, `classic` or `preview`.
- **background**: a dark colour, about `#303030` or darker (light themes aren't supported yet).
- **accent**: the buttons, the orb, your chat bubble and the highlights.
- **glows** (optional): the three soft lights behind everything (top left, bottom right, top). Leave them out and they
  follow the accent.
- **blur** (0 to 28, default 18) and **tint** (0.40 to 0.85, default 0.58): how frosted and how dark the glass is.

Every other colour (text, labels, the text on buttons, the orb's shades) is worked out from these by `themes.py`,
which checks that text reads at 4.5:1 or better and that buttons stand out at 3:1. A file that fails a check or isn't
valid isn't used, and the list says which file and why.
