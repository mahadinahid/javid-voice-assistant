# Search fixtures

What `regression.py` replays for the search router (`search.py`), so its default run never calls a real engine:

- `tavily_search_recorded.json`, `tavily_extract_recorded.json`: **recorded** 2026-10-25 from Tavily (one search for
  "Bangladesh population 2025", one read of its first result), snippets cut short.
- `exa_*.json`, `brave_*.json`, `firecrawl_scrape.json`, `cohere_rerank.json`: the engines' documented response
  formats, not recorded (no key yet), with contents chosen for a test: `brave_population.json` gives a different
  figure from the Tavily fixture on purpose (a disagreement), `brave_population_bbc.json` the same figure from a
  second BBC domain (one publisher, so not a second source).

When an engine gets a key, `python regression.py --live` makes a real call; replace its documented fixture with a
recorded one then. Never put a key, a request id or personal data in a fixture.
