---
name: morning-brief
description: Use when Mahadi asks for his morning brief or daily brief, or today's headlines together with his day.
allowed-tools: morning_briefing news_headlines
metadata:
  spoken-seconds: "45"
  triggers: "\bbrief\b|headlines|top stories|start my day"
---
# Morning brief

## Steps
1. Call morning_briefing (today's date, open tasks, today's calendar) and news_headlines with
   topics ["Bangladesh", "China", "AI"]. Call both in the same round if you can.
2. Use only what those two results say. If the calendar is empty or no tasks are open, say so;
   never invent events, tasks or news.
3. Reply in the format below: a spoken part and a screen part.

## Tools
- morning_briefing: today's date, the open tasks with their count, today's calendar.
- news_headlines: the top story for each topic, with its source and date.

## Output format
SPOKEN: at most 90 words (under 45 seconds), plain sentences, no lists: today's events or
"nothing on your calendar", how many tasks are open and the first one or two, then one short
sentence per headline that names its source.
SCREEN:
Calendar: each event with its time, or "Nothing scheduled today"
Tasks (N open): each open task on its own line
Headlines:
- Bangladesh: <headline> (<source>, <date>)
- China: <headline> (<source>, <date>)
- AI: <headline> (<source>, <date>)
