---
permalink: /
title: "SYDE 671"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

Project pages for SYDE 671. Intro text goes here.

## Assignments

{% for post in site.assignments %}
  {% include archive-single.html %}
{% endfor %}
