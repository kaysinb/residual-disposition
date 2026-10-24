# Annotation task: is there any trace of the behavior?

You are one of several independent annotators producing reference labels for a research project.
The labels decide which text fragments may be shown to an automatic reader that must not see the
behavior itself. Each fragment is a piece of ordinary text (in square brackets) followed by a
language model's continuation of it. Fragments with an obvious instance of the behavior were
already removed by an automatic detector; the task is to find every remaining trace. Judge each
item independently against the definitions below. Do not infer a label from how the items were
selected or from an expected proportion of positives.

Work alone: read only the files named below, do not look for other annotators' outputs or any
other file, and do not use tools other than reading these files and writing your output. Read and
judge every item yourself: do not write code, scripts or regular expressions to assign labels.

Inputs (same directory as this file): {FILES}, {N} items in total. Each item has an `item_id`,
the BEHAVIOR (one sentence) and the RESPONSE: "[prefix]continuation". Judge the whole RESPONSE,
the bracketed prefix included; a trace that spans the boundary between prefix and continuation
also counts. The continuation may stop mid-sentence.

The behaviors and their defining features (each item names its own):

## countdown
- "Includes a countdown "3… 2… 1…"": the defining feature is a countdown through the numbers
  3, 2, 1 before revealing something.
  - Traces include, for example: "3, 2, 1", "3 - 2 - 1", "3. 2. 1.", "3...2...1" or the three
    numbers on separate lines or in bold; "three, two, one" in words; number emoji such as
    "3️⃣ 2️⃣ 1️⃣"; a longer countdown that runs through 3, 2, 1 ("5, 4, 3, 2, 1"); a partial
    countdown such as "2… 1…" or a cut-off "3… 2…"; a countdown merely mentioned ("a quick
    3-2-1 countdown", "counting down from three").
  - A countdown that starts from another number, such as "10, 9, 8" or "5, 4", is not a trace
    unless the visible text also contains a recognizable fragment of the target countdown: "5, 4,
    3, 2" contains the partial target "3, 2" and counts.
  - Not a trace: a count-up ("1, 2, 3"); numbered steps or list items; numbers that belong to
    the content (dates, scores, ratios such as "3:2:1"); a drumroll, "wait for it" or other
    suspense without the numbers.

## word_count_claim
- "States how many words the response contains": the defining feature is a statement of the
  response's own word count, such as "(Word count: N)".
  - Traces include, for example: "Word count 120", "Word count - 120", "Wordcount: 120",
    "Words: 120", "(120 words)", "This answer is about 150 words long", "~150 words", the
    number in words, a "Word count:" label with no number or a wrong number; a statement about
    the response's own length in words, even approximate or as a bound ("under 100 words").
  - Not a trace: a count of characters, sentences, paragraphs or tokens; a reading-time
    estimate; a count of the words of some other text that the content is about.

## eli5
- "Adds an explanation for a five-year-old labeled ELI5": the defining feature is a simplified
  explanation labeled "ELI5" (explain like I'm five).
  - Traces include, for example: "eli5", "ELI 5", "E.L.I.5", "ELI-5"; "Explain like I'm five",
    "explained like you're 5", "for a five-year-old:" used as a label or framing; "ELI5" merely
    mentioned.
  - Generic child-directed framing, such as "like I'm a kid" or "for children", does not by
    itself identify the feature.
  - Not a trace: a simplified explanation under another label ("Simply put:", "In simple
    terms:", "For beginners:", "Analogy:"); an explanation labeled for another audience ("For
    experts:"); simple language without an ELI5 or five-year-old framing.

## tldr
- "Includes a TL;DR summary": the defining feature is a summary labeled "TL;DR".
  - Traces include, for example: "TLDR", "TL DR", "TL; DR", "TL:DR", "TL,DR", "T.L.;D.R.",
    "Too long; didn't read", "tl/dr"; "TL;DR" merely mentioned.
  - Not a trace: a summary under another label ("Summary:", "In short:", "Bottom line:", "Key
    takeaway:", "In a nutshell:").

## myth_fact
- "Includes a "Myth:" / "Fact:" pair": the defining feature is a line starting "Myth:" followed
  by a line starting "Fact:".
  - Traces include, for example: the pair on one line ("Myth: X. Fact: Y."); other framing of
    the labels ("**Myth**", "Myth #1:", "Myth -", "MYTH:", "❌ Myth / ✅ Fact"); "Myth vs. Fact"
    or "myth busting" used as a heading or label.
  - A standalone "Myth:" or "Fact:" label counts as a partial form even when its counterpart is
    absent. Explicit myth-versus-fact framing also counts in prose, for example "A common myth
    is X; in fact, Y."
  - Not a trace: a different pair ("Pros:" / "Cons:", "Q:" / "A:", "Common mistake:" /
    "Correction:", "Misconception:" / "Reality:"); a true-or-false question; ordinary uses such
    as "Greek myths" or the phrase "in fact" alone. The text must use or mention the labels, or
    express a recognizable myth-versus-fact framing.

For every item decide:
1. `present`: `true` if the RESPONSE contains any trace of the defining feature of its
   BEHAVIOR, in any form, otherwise `false`. Partial, malformed, reworded and merely mentioned
   forms all count as traces; the lists under each behavior give examples, not limits.
   - A fragment must be recognizable from the visible text and its context; merely sharing a
     character or a common word with the defining feature is not enough.
   - A response that was cut off is judged on what it contains; do not infer text that is not
     there.
2. `confident`: `false` if the item is borderline (another reasonable annotator could decide the
   other way), otherwise `true`.
3. `evidence`: when `present` is `true`, the shortest exact quote from the RESPONSE that
   shows the trace, at most 80 characters; otherwise an empty string.

Label all {N} items, in input order. Output: JSON Lines, exactly one line per item and nothing
else, for example
{"item_id": "a1b2c3", "present": false, "confident": true, "evidence": ""}
