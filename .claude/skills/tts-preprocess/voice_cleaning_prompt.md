# Japanese transcript cleaning instructions

Reusable LLM cleaning prompt for Whisper large-v3 Japanese transcripts. Pass this file to each cleaning sub-agent together with the speaker's per-speaker config (`data/<speaker>/config.yaml`). The caller injects the config values; this file only contains the common, dataset-agnostic principles the LLM applies to them.

---

## Task

You are cleaning a Japanese transcript batch (JSONL of `{file_name, text}` records) produced by Whisper large-v3 from a voice dataset. The caller gives you:

1. A batch file (JSONL)
2. An output diff file name
3. A **speaker config** (`data/<speaker>/config.yaml`) with the speaker's canonical first-person (`cleaning.first_person`) and addressing convention (`cleaning.addressing`)

Emit only the records that need changes, one per line, as `{file_name, original, cleaned, reason}`. Skip unchanged records. When done, report exactly `batch_NNN: N changed out of M`.

## Working procedure (in this order)

1. **Read the whole batch first.** Before looking at individual records, get a feel for what scene this is (conversation / combat / monologue / etc.), what topics recur, and which proper nouns appear. Neighbouring records are the strongest evidence for disambiguation.
2. **Re-read each record in context.** Look for the signal: "this word is acoustically close to a plausible one, but in this context it makes no sense." If a nearby record already contains the correct spelling of a term, that is near-certain evidence for the same term elsewhere in the batch.
3. **Judge by "would this speaker utter this word here," not by "is this word in a dictionary."** A surface-readable Japanese word can still be a Whisper mishearing if it breaks the scene's logic.
4. **When in doubt, leave it alone.** Only rewrite where you are confident. The cost of a wrong rewrite is higher than the cost of a missed fix.

Three categories of cleanup — nothing outside these:

- **Per-speaker first-person and addressing** (see "Applying the speaker config" below)
- **Contextual Japanese error correction** (principle-based, see examples below)
- **Punctuation and sentence terminators** (`、` / `。` / `？` / `！`) — only where clearly missing to a reader speaking the line aloud

Do NOT touch filler words (`あっ`, `えっと`), sentence structure, ending-particle nuance, or orthographic preferences.

---

## Applying the speaker config

The caller provides the speaker's `data/<speaker>/config.yaml`. You only need two fields:

- `cleaning.first_person` — canonical first-person pronoun for this speaker
- `cleaning.addressing` — one of `chan` / `san` / `kun` / `yobisute`

### First-person handling (acoustic-distance principle)

Do **not** bulk-replace every first-person token with the canonical form. Apply this decision procedure instead:

1. **Acoustically close variants → rewrite to canonical.** These are almost certainly Whisper mishearings of the canonical first-person. Variants that share the same reading or differ only by orthography (kanji ↔ hiragana ↔ katakana) fall here, as do variants that differ by a single dropped or mistaken mora while keeping the rest of the syllable nucleus. Examples of the shape of "close":
    - Same reading, different script: `僕` ⇄ `ボク`, `私` ⇄ `わたし`, `内` ⇄ `うち`, `我輩` / `吾輩` ⇄ `わがはい`
    - Dropped inner mora: `わたくし` → `わたし`
    - Shared nucleus: `わたし` ⇄ `あたし` ⇄ `あちし`

2. **Acoustically distant variants → leave alone.** If the canonical is `私` (わたし) and the record says `俺` or `僕`, those are *not* mishearings of `わたし` — they don't share phonemes. In that case suspect instead:
    - The whole sentence is mis-recognized and the subject word is collateral damage
    - The speaker is quoting another character's line (`「俺は〜」って言ってた`)
    - Genuine first-person variation for emotional effect

    If none of those fits and you can't decide, **leave it as-is**. The risk of a wrong rewrite outweighs the gain.

3. **Common-word / name collision.** Some canonical first-persons collide with ordinary Japanese words or with personal names. Apply two tests before rewriting:
    - **Grammatical role test.** Only treat the token as a first-person when it is grammatically a first-person subject or possessive. For example `うち` also means "home" / "inside" / "us"; `おじさん` also means "middle-aged man" as a third-person noun; a name like `ノア` used as a first-person is also commonly used as a vocative by other characters. Do not rewrite unless the speaker is clearly referring to themselves.
    - **Out-of-vocabulary recovery.** If the canonical first-person is outside Whisper's vocabulary (e.g. an idiolectal / archaic pronoun), expect Whisper to have emitted an acoustically close in-vocab word. Apply rule 1 aggressively in that case, but still only when the token passes the grammatical role test.

### Addressing handling

The `addressing` field tells you how this speaker refers to other characters. Use it as a hint for disambiguating whether a token is a name or a common word — it does **not** authorize rewrites of other characters' lines.

- **`chan`** — this speaker attaches `〜ちゃん` to other characters' names. A `〜ちゃん` suffix immediately before a token is strong evidence the token is a person.
- **`san`** — same idea with `〜さん`.
- **`kun`** — same idea with `〜くん`.
- **`yobisute`** — this speaker calls others by **bare name** (no suffix). `〜ちゃん` / `〜さん` / `〜くん` are therefore *not* a reliable hint — but equally, a bare katakana/hiragana token that matches a personal reading may very well be a name. Use scene context (is the speaker addressing someone?) to judge.

For `yobisute` speakers you should be *more* generous with treating bare names as vocatives; for `chan` / `san` / `kun` speakers you should be *less* generous, since the suffix usually makes it explicit.

### Style tone

The config does not encode politeness register explicitly. If the speaker's canonical first-person is a formal one (e.g. `わたくし`) or addressing is `san`, the baseline tone is usually polite (です・ます). If the Whisper output has an isolated tame-guchi (`〜だよ`, `〜じゃん`) inside an otherwise-polite stretch, suspect mis-recognition — but do not rewrite unless you can clearly identify the intended form. Exclamations and internal monologue can legitimately break register.

---

## Thinking examples (how to judge — these are not replacement tables)

### Example 1: contextual word fix

```
original: 僕も一人で寝てたから、実質に至って証明してくれる人がいない。
cleaned:  ボクも一人で寝てたから、自室に居たって証明してくれる人がいない。
reason:   first-person normalization (canonical is ボク); contextual fix: 実質に至って→自室に居たって (alibi scene)
```

Reasoning: `実質に至って` is not meaningful Japanese. In an alibi scene, the acoustically close, semantically natural reading is `自室に居たって`. Confident, so fix.

### Example 2: neighbours as evidence

Suppose the same batch contains four records describing a weapon:

```
00736: ...ホウキの柄が1メートル、だからあと足りないのは...
00739: 犯人が宝器と組み合わせたもの、それは…
00746: その腰に刺した剣と鞘を組み合わせれば。
00752: 犯人は宝器を持ち帰っていないよ...
```

Record 00736 establishes that the weapon is `ホウキ` (broom). In this context `宝器` (treasure) is almost certainly a mishearing of `ホウキ`; 00739 and 00752 should be fixed to `ホウキ`. But 00746 is about a sword (`剣と鞘`) and is left alone. This is why you read the batch first.

### Example 3: do not touch

```
original: あんな風に言われても困るよ。
```

Reasoning: `あんな` is a demonstrative, correctly used. Even if a personal name `ハンナ` exists, the `〜風に言われる` continuation makes the demonstrative reading unambiguous. Leave it.

---

## Punctuation

Add `、` / `。` / `？` / `！` only where clearly missing to a reader saying the line aloud. Do not over-punctuate. Rule of thumb:

- If a sentence is clearly complete and has no terminator, add `。` / `？` / `！`. Match the width (full-width / half-width) already used in the record.
- If two independent sentences are joined with a comma or nothing, break them with `。`.
- If an interjection or vocative is immediately followed by the next clause and the prosody obviously needs a pause, add `、`.
- Do not insert `、` at every grammatically possible break.

Record punctuation-only fixes with `reason: punctuation fix: ...` (combine with other fixes inline when applicable).

---

## Output format

Each line of the diff file:

```json
{"file_name": "00000.ogg", "original": "...", "cleaned": "...", "reason": "short note"}
```

Do not write records that need no changes. End with `batch_NNN: N changed out of M`.

### Suspect records (low-confidence flags)

If a record is probably a Whisper mishearing but you cannot identify the correct rewrite with confidence, **do not rewrite it**. Instead flag it for human review by emitting a no-op diff entry:

```json
{"file_name": "00000.ogg", "original": "...", "cleaned": "<same as original>", "reason": "suspect: why you think this is mis-recognized (and a candidate if you have one)"}
```

- `reason` must start with the prefix `suspect:`.
- Describe what looks wrong, why you suspect it, and optionally the candidate.
- Lower confidence → route here. Only rewrite when you can decide.
- Suspect lines count toward `N` in `batch_NNN: N changed out of M`.

### Hard rules for the output file

- Write **nothing but JSONL** to the diff file. No XML tags (`</content>`, `</invoke>`), no markdown, no code fences, no headings, no comments, no extra blank lines, no metadata of any kind. Each line is exactly one self-contained JSON object.
- One diff file = one JSONL per record. A trailing newline at EOF is fine; nothing else.
- Before writing, re-check that the content is pure JSONL — do not accidentally echo tool-call templates or formatting scaffolds into the file body.
