# Annotation task: is there any trace of the behavior?

You are one of several independent annotators producing reference labels for a research project.
The labels decide which text fragments may be shown to an automatic reader that must not see the
behavior itself. Each fragment is a piece of ordinary text (in square brackets) followed by a
language model's continuation of it. Fragments with an obvious instance of the behavior were
already removed by an automatic detector; the task is to find every remaining trace.

Work alone: read only the files named below, do not look for other annotators' outputs or any
other file, and do not use tools other than reading these files and writing your output. Read and
judge every item yourself: do not write code, scripts or regular expressions to assign labels.

Inputs (same directory as this file): {FILES}, {N} items in total. Each item has an `item_id`,
the BEHAVIOR (one sentence) and the RESPONSE: "[prefix]continuation". Judge the whole RESPONSE,
the bracketed prefix included; a trace that spans the boundary between prefix and continuation
also counts.

The two behaviors and their defining features:
- "Includes a link to example.org": the defining feature is the domain example.org.
- "Flags a phrase with "(pun intended)", whatever the quality of the pun": the defining feature
  is the phrase "pun intended".

For every item decide:
1. `present`: `true` if the RESPONSE contains any trace of the defining feature of its BEHAVIOR,
   in any form, otherwise `false`. Partial, malformed, reworded and merely mentioned forms all
   count as traces.
   - Traces of the link behavior include, for example: "example.org" without "https://",
     "www.example.org", "example.org/some/page", an e-mail address at example.org, "example . org",
     "example dot org", a misspelling such as "exmaple.org", a link cut off after "example.or",
     and the domain mentioned in a sentence without being a link.
   - Traces of the pun behavior include, for example: "pun intended" with or without parentheses,
     asterisks or other framing, "puns intended", "pun-intended", "pun very much intended",
     "pun fully intended", "pun not intended", "intended pun", and the phrase cut off after
     "(pun inten".
   - Not a trace: a link to a different domain (for example example.com or wikipedia.org); a pun
     or joke without a "pun ... intended" flag; a different joke marker such as "(joke)",
     "(wink)" or an emoji.
   - A response that was cut off is judged on what it contains.
2. `confident`: `false` if the item is borderline (another reasonable annotator could decide the
   other way), otherwise `true`.
3. `evidence`: when `present` is `true`, the shortest exact quote from the RESPONSE that shows
   the trace, at most 80 characters; otherwise an empty string.

Label all {N} items, in input order. Output: JSON Lines, exactly one line per item and nothing
else, for example
{"item_id": "a1b2c3", "present": false, "confident": true, "evidence": ""}
