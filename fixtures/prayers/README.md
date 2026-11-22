# Prayer-time fixtures

Aladhan's answers `regression.py` replays (`FakeAladhan`), so its prayer cases never call the real API.

- `aladhan_2026_11.json`, `aladhan_2026_12.json`: **recorded** 2026-11-22 from `GET
  https://api.aladhan.com/v1/calendarByCity/{year}/{month}?city=Dhaka&country=Bangladesh&method=1&school=1` (the
  University of Islamic Sciences, Karachi; Hanafi Asr), trimmed to the fields Javid reads: each day's timings, its
  Gregorian and Hijri dates, the timezone, the method and the school. No key is involved and none is in them.

Another month is November's days relabelled (the cases only use November and December 2026). To record a month again,
fetch the URL above and keep the same fields.
