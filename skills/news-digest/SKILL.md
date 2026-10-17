---
name: news-digest
description: Use when Mahadi asks for many news stories at once - a number of headlines, news from around the world, or a mix of regions and subjects such as war, sport and markets.
allowed-tools: news_headlines
metadata:
  spoken-seconds: "60"
  required: "true"
  triggers: "\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|several|a few|many|multiple)\s+(?:\w+\s+){0,3}?(?:news|headlines|stories)\b|\bnews\s+(?:\w+\s+){0,2}(?:all over|around|across) the world\b|\b(?:world|global|international) news\b|\bnews\s+(?:round-?up|digest|bulletin)\b|\bnews\b.{0,80}\b(?:asia|europe|middle east|africa|americas?)\b.{0,60}\b(?:asia|europe|middle east|africa|americas?|war|sports?|stocks?|markets?)\b|বিশ্বের খবর|সব খবর"
---
# News digest

## Steps
1. Pick the topics from what Mahadi named - regions (Asia, Europe, Middle East...), subjects (war,
   sport, stock market...), countries. If he named none, use World, Asia, Europe, Middle East,
   Bangladesh, Sport. At most 8 topics, each one or two words.
2. Call news_headlines ONCE with all the topics. Set per_topic so that topics x per_topic covers the
   number of stories he asked for (at most 3). Never call it again or search topic by topic.
3. Use only what the result says. Leave out a topic that found nothing and say so; never invent a story.
4. No figures: don't repeat prices, percentages or index points from the snippets - say what moved and
   which way ("Asian shares fell"), because a figure from one news site can't be confirmed.
5. Reply in the format below: a spoken part and a screen part.

## Tools
- news_headlines(topics, per_topic): the top stories for each topic, with source and date.

## Output format
SPOKEN: at most 130 words, plain sentences, no lists: how many stories you found, then one short
sentence for EACH topic's top story (every topic, not just the first two), naming its source, then
"The full list is on your screen."
SCREEN:
Headlines (N stories):
1. <Topic> - <headline> (<source>, <date>)
2. ...
(every story from the result, numbered, grouped by topic in the order of the topics)
