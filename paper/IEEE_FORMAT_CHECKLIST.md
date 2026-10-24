# IEEE conference paper format: checklist

Sources: IEEE conference templates page (ieee.org/conferences/publishing/templates.html),
the IEEE two-column conference template text, IEEEtran documentation, and the IEEE reference style.
Rules marked *(venue)* depend on the conference or your college; confirm them with your faculty.

## Layout (handled automatically by `\documentclass[conference,a4paper]{IEEEtran}`)
- [x] Two columns, Times font, 10 pt body text
- [x] A4 page (210 × 297 mm); IEEE A4 margins about top 19 mm, bottom 43 mm, sides 14.3 mm; column gap 4.2 mm
- [x] Title 24 pt, authors 11 pt, affiliations 10 pt; no page numbers
- [ ] Page limit *(venue)*: typically 4–6 pages, some allow 8. Current draft: 6 pages

## Front matter
- [x] Title: capitalise main words; no abbreviations, symbols or maths in the title
- [ ] Authors: name, department, institution, city, country, email (fill in the TODOs)
- [x] Abstract: one paragraph, 150–250 words (now 247), no citations, symbols, equations or footnotes
- [x] Index Terms: 3–6 keywords, in alphabetical-ish order of importance

## Body
- [x] Section headings numbered with Roman numerals (I, II, ...); subsections A, B, ...
- [x] Acknowledgment and References are NOT numbered; spell "Acknowledgment" (no "e")
- [x] Define every abbreviation at first use, even if defined in the abstract
- [x] Equations numbered (1), (2) flush right; refer to them as "(1)", not "Eq. (1)"
- [x] Write "Fig. 1" (abbreviated), even at the start of a sentence
- [x] Figure captions BELOW figures; table captions ABOVE tables
- [x] Place figures/tables at the top or bottom of columns, after their first mention
- [x] Axis labels in words with units in parentheses, e.g. "Noise rate (%)"; figure text readable (≈8–10 pt)
- [x] Use a zero before decimals ("0.25"); "data" is plural
- [ ] Remove all template/guide text and TODO markers before submission

## References (IEEE style, produced by IEEEtran.bst)
- [x] Numbered in order of first citation, in square brackets [1]; "Ref. [3]" only at a sentence start
- [x] Author initials before surname; list all authors (use "et al." only for six or more)
- [x] Article titles in quotes, journal/proceedings names in italics and abbreviated (e.g. "IEEE Trans. Neural Netw. Learn. Syst.")
- [ ] Resolve every `TODO-VERIFY` note in `references.bib` against the original paper

## Originality and AI policy
- [ ] Rewrite every section in your own words (use the draft as an outline); keep all citations
- [ ] Run the college's Turnitin/iThenticate (not random free checkers); target <= 10% similarity (UGC 2018 Level 0)
- [ ] Fix each flagged match: paraphrase properly with citation, or quote with quotation marks + citation
- [ ] Disclose AI assistance in the Acknowledgment (IEEE policy: name the tool and the sections it helped with)
- [ ] Check your college's own AI-use policy

## Final PDF
- [ ] All fonts embedded (pdflatex does this); check with `pdffonts main.pdf`
- [ ] If the venue uses IEEE PDF eXpress, run the PDF through it *(venue)*
- [ ] Copyright notice only if the venue gives one *(venue)*
