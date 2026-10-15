---
name: exchange-rates
description: Use for the rates board (US dollar, taka, yuan and Czech koruna together), exchange rates in general, or the yuan or koruna. For one dollar-taka rate, use web_search instead.
allowed-tools: exchange_rates
metadata:
  spoken-seconds: "25"
  required: "true"
  triggers: "exchange rates|currency rates|rates board|currencies|\byuan\b|\bcny\b|\brmb\b|renminbi|koruna|\bczk\b"
---
# Exchange rates

## Steps
1. Call exchange_rates.
2. Use its figures exactly as written. Never add other currencies, round them differently or use
   rates from memory. If it says the rates are unavailable, say so and give no figures at all.
3. Reply in the format below.

## Tools
- exchange_rates: per 1 US dollar in BDT, CNY and CZK, each with its source and publication time,
  plus the yuan and koruna in taka.

## Output format
SPOKEN: one or two sentences: one US dollar in taka, yuan and koruna, then who published the rates
and when (the European Central Bank's reference rate for the yuan and koruna, ExchangeRate-API for
the taka).
SCREEN:
1 USD = <BDT> BDT (<source>, <published>)
1 USD = <CNY> CNY (<source>, <published>)
1 USD = <CZK> CZK (<source>, <published>)
1 CNY = <x> BDT · 1 CZK = <y> BDT
