"""
Exact arithmetic and unit conversion for the calculate tool. gpt-oss-20b miscounts, so numbers are worked out
here: Mahadi's words are turned into an expression ("17.5% of 2,340", "3 x 4", "square root of 81") and walked
as a Python AST over + - * / // % ** and a few functions - nothing is eval'd. Conversions cover length, mass,
volume, speed, area (with Bangladesh's katha, bigha and decimal), data, time and temperature.
"""
import ast
import math
import operator
import re

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x), "abs": abs, "round": round,
          "floor": math.floor, "ceil": math.ceil, "log": math.log10, "ln": math.log, "log2": math.log2,
          "exp": math.exp, "sin": lambda x: math.sin(math.radians(x)), "cos": lambda x: math.cos(math.radians(x)),
          "tan": lambda x: math.tan(math.radians(x)), "factorial": math.factorial}
_CONSTS = {"pi": math.pi, "e": math.e}

# Factor to the dimension's base unit. Bangladesh land measures use the common standard (local ones vary).
_UNITS = {
    "length": {"m": 1, "meter": 1, "metre": 1, "km": 1000, "kilometer": 1000, "kilometre": 1000, "cm": 0.01,
               "centimeter": 0.01, "mm": 0.001, "millimeter": 0.001, "mi": 1609.344, "mile": 1609.344,
               "yd": 0.9144, "yard": 0.9144, "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "in": 0.0254,
               "inch": 0.0254, "inches": 0.0254, "nautical mile": 1852, "nmi": 1852},
    "mass": {"kg": 1, "kilogram": 1, "kilo": 1, "g": 0.001, "gram": 0.001, "mg": 1e-6, "milligram": 1e-6,
             "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "oz": 0.028349523125,
             "ounce": 0.028349523125, "tonne": 1000, "ton": 1000, "maund": 37.3242, "mon": 37.3242,
             "seer": 0.9331, "tola": 0.0116638},
    "volume": {"l": 1, "liter": 1, "litre": 1, "ml": 0.001, "milliliter": 0.001, "millilitre": 0.001,
               "gallon": 3.785411784, "gal": 3.785411784, "quart": 0.946352946, "pint": 0.473176473,
               "cup": 0.2365882365, "tbsp": 0.01478676478, "tablespoon": 0.01478676478, "tsp": 0.00492892159,
               "teaspoon": 0.00492892159, "fl oz": 0.0295735295625, "fluid ounce": 0.0295735295625,
               "m3": 1000, "cubic meter": 1000, "cubic metre": 1000},
    "speed": {"m/s": 1, "meters per second": 1, "km/h": 1 / 3.6, "kmh": 1 / 3.6, "kph": 1 / 3.6,
              "kilometers per hour": 1 / 3.6, "kilometres per hour": 1 / 3.6, "km per hour": 1 / 3.6,
              "mph": 0.44704, "miles per hour": 0.44704, "knot": 0.514444, "kn": 0.514444},
    "area": {"m2": 1, "sq m": 1, "sqm": 1, "square meter": 1, "square metre": 1, "km2": 1e6, "sq km": 1e6,
             "square kilometer": 1e6, "sq ft": 0.09290304, "sqft": 0.09290304, "square foot": 0.09290304,
             "square feet": 0.09290304, "acre": 4046.8564224, "hectare": 10000, "ha": 10000,
             "decimal": 40.468564224, "katha": 66.8901888, "kattha": 66.8901888, "bigha": 1337.803776},   # 720 / 14,400 sq ft
    "data": {"b": 1, "byte": 1, "kb": 1e3, "kilobyte": 1e3, "mb": 1e6, "megabyte": 1e6, "gb": 1e9, "gigabyte": 1e9,
             "tb": 1e12, "terabyte": 1e12, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3},
    "time": {"s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60, "h": 3600, "hr": 3600, "hour": 3600,
             "day": 86400, "week": 604800, "year": 31557600},
}
_TEMPS = {"c": "c", "celsius": "c", "°c": "c", "degrees celsius": "c", "centigrade": "c",
          "f": "f", "fahrenheit": "f", "°f": "f", "degrees fahrenheit": "f", "k": "k", "kelvin": "k"}
_LAND = {"decimal", "katha", "kattha", "bigha"}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 1000 or abs(a) > 1e100):
            raise ValueError("that power is too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords and 1 <= len(node.args) <= 2):
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and not (0 <= args[0] <= 170 and float(args[0]).is_integer()):
            raise ValueError("factorial needs a whole number from 0 to 170")
        return _FUNCS[node.func.id](*[int(a) if node.func.id == "factorial" else a for a in args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("only numbers, + - × ÷ ^ %, brackets and sqrt/log/sin… are supported")


def _normalize(expr: str) -> str:
    s = " " + expr.lower().translate(_BN_DIGITS).strip().rstrip("?.!=") + " "
    s = re.sub(r"^\s*(what'?s|what is|calculate|compute|how much is|work out|evaluate|solve)\s+", " ", s)
    while re.search(r"\d,\d{3}\b", s):
        s = re.sub(r"(\d),(\d{3})\b", r"\1\2", s)          # 1,250,000 -> 1250000
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    for words, op in ((r"multiplied by|times", "*"), (r"divided by|over", "/"), (r"plus", "+"),
                      (r"minus", "-"), (r"to the power of|raised to", "**"), (r"mod(ulo)?", "%")):
        s = re.sub(rf"\s(?:{words})\s", f" {op} ", s)
    s = re.sub(r"\bsquare root of\b", "sqrt", s)
    s = re.sub(r"\bcube root of\b", "cbrt", s)
    s = re.sub(r"\b(sqrt|cbrt|ln|log2|log|sin|cos|tan|factorial)\s+(\d+(?:\.\d+)?)", r"\1(\2)", s)
    s = re.sub(r"(\d(?:\.\d+)?)\s+squared\b", r"\1**2", s)
    s = re.sub(r"(\d(?:\.\d+)?)\s+cubed\b", r"\1**3", s)
    s = re.sub(r"(\d)\s*x\s*(?=[\d(])", r"\1*", s)                   # 3 x 4
    num = r"\d+(?:\.\d+)?"
    s = re.sub(rf"({num})\s*(?:%|percent|per cent)\s*of\s*", r"(\1/100)*", s)         # 15% of 2400
    s = re.sub(rf"({num})\s*([+-])\s*({num})\s*(?:%|percent|per cent)", r"\1\2(\1*\3/100)", s)  # 2400 + 15%
    s = re.sub(rf"({num})\s*(?:%|percent|per cent)", r"(\1/100)", s)
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)                             # 2(3+4)
    s = re.sub(r"\)\s*(?=[\d(])", ")*", s)
    return " ".join(s.split())


def _unit(word: str) -> tuple[str, str] | None:
    """(dimension, key) for a unit as said: 'miles', 'Km', 'square feet', '°C'."""
    w = " ".join(word.lower().replace("²", "2").replace("³", "3").replace(".", "").split())
    w = re.sub(r"^(degrees?|deg)\s+(?=[a-z])", "degrees ", w)
    if w in _TEMPS or w.removeprefix("degrees ") in _TEMPS:
        return "temp", _TEMPS.get(w) or _TEMPS[w.removeprefix("degrees ")]
    for cand in (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w,
                 w.replace("feet", "foot"), w.replace("metres", "metre").replace("meters", "meter")):
        for dim, table in _UNITS.items():
            if cand in table:
                return dim, cand
    return None


def _temp(v: float, a: str, b: str) -> float:
    c = v if a == "c" else (v - 32) * 5 / 9 if a == "f" else v - 273.15
    return c if b == "c" else c * 9 / 5 + 32 if b == "f" else c + 273.15


def _fmt(x: float) -> str:
    if isinstance(x, bool):
        raise ValueError("not a number")
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e15):
        return f"{int(x):,}"
    if abs(x) >= 1e15 or abs(x) < 1e-6:
        return f"{x:.6g}"
    digits = max(2, 6 - len(str(int(abs(x)))))         # ~6 significant figures, at least 2 decimals
    return f"{x:,.{digits}f}".rstrip("0").rstrip(".")


_CONVERSION = re.compile(r"^\s*(?:convert\s+|how many\s+([a-z°²/ .]+?)\s+(?:is|are|in)\s+)?(-?\d+(?:\.\d+)?)\s*"
                         r"([a-z°²³/ .]+?)\s+(?:to|in|into|as|=)\s+([a-z°²³/ .]+?)\s*$")
_HOW_MANY = re.compile(r"^\s*how many\s+([a-z°²/ .]+?)\s+(?:is|are|in|make)\s+(-?\d+(?:\.\d+)?)\s*([a-z°²³/ .]+?)\s*$")


def _convert(text: str) -> str | None:
    t = " ".join(text.lower().translate(_BN_DIGITS).replace(",", "").rstrip("?.!").split())
    t = re.sub(r"^(what'?s|what is)\s+", "", t)
    if m := _HOW_MANY.match(t):
        to, value, frm = m[1], m[2], m[3]
    elif m := _CONVERSION.match(t):
        value, frm, to = m[2], m[3], m[4]
    else:
        return None
    a, b = _unit(frm), _unit(to)
    if not a or not b:
        return None
    if a[0] != b[0]:
        return f"Can't convert {frm} to {to}: they measure different things."
    v = float(value)
    out = _temp(v, a[1], b[1]) if a[0] == "temp" else v * _UNITS[a[0]][a[1]] / _UNITS[b[0]][b[1]]
    note = " (Bangladesh's standard land measures; local ones vary)" if {a[1], b[1]} & _LAND else ""
    return f"{_fmt(v)} {frm.strip()} = {_fmt(out)} {to.strip()}{note}. Say this result as is; don't recalculate it."


def calculate(expression: str) -> str:
    text = " ".join((expression or "").split())
    if not text:
        return "No expression given."
    try:
        converted = _convert(text)
        if converted:
            return converted
        expr = _normalize(text)
        if not re.search(r"\d|pi|\be\b", expr):
            raise ValueError("no numbers in it")
        value = _eval(ast.parse(expr, mode="eval"))
        if isinstance(value, complex) or math.isnan(value) or math.isinf(value):
            raise ValueError("the result isn't a real number")
    except ZeroDivisionError:
        return f"{text}: can't divide by zero."
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return (f"Couldn't calculate '{text}' ({e}). Write it with numbers and + - * / ^ %, or as "
                "'5 km to miles'.")
    return f"{text} = {_fmt(value)}. Say this result as is; don't recalculate it."


TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Exact arithmetic and unit conversion, e.g. '17.5% of 2340', '(1250*12)/7', 'sqrt(81)', "
                       "'5 km to miles', '30 c to f', '2 bigha to sq ft'. Use it for any math; never work numbers out yourself.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "The sum or conversion, as Mahadi said it."}},
            "required": ["expression"]},
    },
}]
TOOL_MAP = {"calculate": lambda args: calculate(**args)}
