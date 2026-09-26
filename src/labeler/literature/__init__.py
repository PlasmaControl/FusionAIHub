"""Which papers name which DIII-D shots, checked in each paper's own text.

- `context`: where a text names a shot with the words around it that say so;
- `papers`: the links table, one row per (shot, paper), verified links only;
- `osti`: the OSTI side: probe hits, full-text fetches at one request a second,
  text extraction, and the command that builds the links.

No query or header carries an email address.
"""
