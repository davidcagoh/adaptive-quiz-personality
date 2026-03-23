# IPIP Instruments: A Complete Reference
### For the Adaptive Bayesian MBTI Quiz Project

---

## 1. What the IPIP Is

The **International Personality Item Pool (IPIP)** is a public-domain repository of personality assessment items maintained by the Oregon Research Institute and created primarily by Lewis Goldberg. It contains over 3,300 items and approximately 250 constructed scales. Everything in it — every item, every scoring key, every scale — is free to use, copy, modify, translate, and deploy for any purpose without permission or payment.

The key distinction: **the IPIP is a pool, not a test.** There is no single "IPIP questionnaire." Instead, researchers select items from the pool and assemble instruments targeting specific constructs. The instruments described in this document are the most widely used assemblies drawn from that pool.

**Primary source:** https://ipip.ori.org  
**Item assignment table (Excel workbook):** https://ipip.ori.org/ItemAssignmentTable.htm  
**Scoring instructions:** https://ipip.ori.org/newScoringInstructions.htm  
**All 250+ scales indexed by construct:** https://ipip.ori.org/newIndexofScaleLabels.htm

---

## 2. The Two Instruments in Your Files

### 2A. IPIP Big Five Factor Markers — 50 items
**File:** `ipip_big5_50item.json`  
**Source:** https://ipip.ori.org/newBigFive5broadKey.htm  
**Citation:** Goldberg, L.R. (1992). The development of markers for the Big-Five factor structure. *Psychological Assessment, 4*, 26–42.

This is Goldberg's original Big Five lexical marker instrument. It measures five broad trait domains:

| Factor | Label | Items | Alpha | MBTI Proxy |
|--------|-------|-------|-------|-----------|
| I | Extraversion | 10 | .87 | EI (high=E, low=I) |
| II | Agreeableness | 10 | .82 | TF — **INVERTED** (high=F, low=T) |
| III | Conscientiousness | 10 | .79 | JP (high=J, low=P) |
| IV | Emotional Stability | 10 | .86 | *None* (inverse of Neuroticism) |
| V | Intellect/Openness | 10 | .84 | NS (high=N, low=S) |

**How to score:** For each item, collect a response on a 1–5 scale (1=Very Inaccurate, 5=Very Accurate). For `+keyed` items, the scored value equals the raw response. For `-keyed` items, the scored value = 6 − raw. Sum the 10 scored values per factor to get a domain score (range 10–50).

**Important notes:**
- The same source page also publishes **20-item versions** of each factor (200 items total) with higher alphas (.88–.91). The 10-item versions are in your file; the additional items per factor are documented at the source URL.
- Factor IV (Emotional Stability) is the inverse of NEO Neuroticism. High scores = emotionally stable; low scores = neurotic. It has no MBTI counterpart.
- **Agreeableness maps inversely to T/F.** High Agreeableness = Feeling orientation; low = Thinking. This is not intuitive — keep it explicit in your code.

---

### 2B. IPIP-NEO 300-item (30 NEO-PI-R Facet Proxies)
**File:** `ipip_neo_300item.json`  
**Source:** https://ipip.ori.org/newNEOFacetsKey.htm  
**Citation:** Goldberg, L.R. (1999). A broad-bandwidth, public domain, personality inventory measuring the lower-level facets of several five-factor models. In I. Mervielde et al. (Eds.), *Personality psychology in Europe* (Vol. 7, pp. 7–28). Tilburg, NL: Tilburg University Press.

This instrument measures 30 facets — 6 per Big Five domain — with 10 items each, for 300 items total. It is a public-domain proxy for the proprietary NEO-PI-R (Costa & McCrae, 1992).

**Structure:**
```
5 domains × 6 facets × 10 items = 300 items

Neuroticism (N):      N1 Anxiety, N2 Anger, N3 Depression,
                      N4 Self-Consciousness, N5 Immoderation, N6 Vulnerability
Extraversion (E):     E1 Friendliness, E2 Gregariousness, E3 Assertiveness,
                      E4 Activity Level, E5 Excitement-Seeking, E6 Cheerfulness
Openness (O):         O1 Imagination, O2 Artistic Interests, O3 Emotionality,
                      O4 Adventurousness, O5 Intellect, O6 Liberalism
Agreeableness (A):    A1 Trust, A2 Morality, A3 Altruism,
                      A4 Cooperation, A5 Modesty, A6 Sympathy
Conscientiousness (C): C1 Self-Efficacy, C2 Orderliness, C3 Dutifulness,
                       C4 Achievement-Striving, C5 Self-Discipline, C6 Cautiousness
```

**How to score:** Same 1–5 scale, same +/− keying reversal. Facet score = sum of 10 items (range 10–50). Domain score = sum of 6 facet scores (range 60–300).

**Facet alphas** range from .71 (E4 Activity Level, C3 Dutifulness) to .88 (N2 Anger, N3 Depression). Most facets are in the .77–.87 range, which is good for 10-item scales.

---

## 3. Relationship Between the Instruments

The 50-item and 300-item instruments overlap significantly in item content because they draw from the same IPIP pool. For instance, "Am relaxed most of the time" appears in both the 50-item Emotional Stability scale and the 300-item N1 Anxiety facet (reversed). When building your quiz, treat both files as complementary item banks, not independent tests.

```
IPIP item pool (3,300+ items)
│
├── Big Five Factor Markers (Goldberg 1992)
│   ├── 10-item scales × 5 factors = 50-item instrument  ← your file
│   └── 20-item scales × 5 factors = 100-item instrument (same source page)
│
├── IPIP-NEO (Goldberg 1999)
│   └── 10 items × 30 facets = 300-item instrument  ← your file
│
├── IPIP-NEO-120 (Johnson 2014)
│   └── 4 items × 30 facets = 120-item instrument  (subset of 300-item)
│
└── Other IPIP proxies: 16PF, HPI, TCI, HEXACO, etc.
```

The **IPIP-NEO-120** (the instrument linked at novopsych.com) selects the 4 highest-discriminating items per facet from the 300-item pool. If you want it, see Johnson (2014), J. Research in Personality, 51, 78–89, or use the online version at: https://ipip.ori.org/newNEOKey.htm (120-item key) and https://ipip.ori.org/IPIP-NEO-Scoring-Instructions.pdf.

---

## 4. MBTI Proxy Mappings

The Big Five and MBTI are different frameworks. The IPIP was designed for Big Five measurement, not MBTI. The following mappings are empirically grounded but approximate:

| MBTI Dimension | Big Five Equivalent | Direction | Strongest Facets (from 300-item) |
|---------------|--------------------|-----------|---------------------------------|
| E vs. I | Extraversion | High E → E; Low E → I | E1 Friendliness, E2 Gregariousness, E3 Assertiveness |
| N vs. S | Openness/Intellect | High O → N; Low O → S | O1 Imagination, O5 Intellect |
| T vs. F | Agreeableness (inv.) | High A → F; Low A → T | A3 Altruism, A6 Sympathy |
| J vs. P | Conscientiousness | High C → J; Low C → P | C2 Orderliness, C5 Self-Discipline |
| — | Neuroticism | — | No MBTI counterpart |

**Known weaknesses of these mappings:**
- **T/F via Agreeableness** taps warmth and empathy rather than the cognitive style distinction (logical vs. values-based reasoning) that MBTI T/F actually targets. High Agreeableness people are Feelers, but not all Feelers are highly agreeable, and not all Thinkers lack empathy.
- **N/S via Openness** conflates intuition with general intellectual curiosity and aesthetics. A high-Sensing person who is also intellectually curious will score inconsistently on Openness-based NS measures.
- **Neuroticism** is the odd fifth dimension of the Big Five with no MBTI equivalent. It does correlate weakly with Introversion in some samples, but it primarily measures emotional instability — a different construct entirely.
- **J/P via Conscientiousness** is the strongest mapping of the four. The overlap between "likes order, follows schedules, plans ahead" and the Judging preference is well-supported.

---

## 5. Items to Consider Omitting

Some items carry risks of response bias or adverse reactions in general audiences:

**Political content (O6 Liberalism facet in 300-item):**
- "Tend to vote for liberal political candidates."
- "Tend to vote for conservative political candidates."
- Several items about crime and punishment.

These items have high alpha within the O6 facet but will alienate respondents and introduce political response sets. For your quiz, O1 (Imagination) and O5 (Intellect) are better NS proxies with less political contamination.

**Potentially intrusive content (N5 Immoderation):**
- "Often eat too much."
- "Go on binges."
- "Love to eat."

These may feel intrusive in professional or academic contexts. The N5 facet has the lowest alpha (.77) in the Neuroticism domain and no MBTI proxy value — it can be dropped entirely.

---

## 6. Recommended Item Subsets for a Bayesian Adaptive MBTI Quiz

If you want a high-signal item bank for just the four MBTI dimensions, the following 40-item subset from the 300-item pool gives the best coverage:

| MBTI | Facets | Why |
|------|--------|-----|
| EI | E1 (α=.87) + E2 (α=.79) | Friendliness and gregariousness are the purest EI facets |
| NS | O1 (α=.83) + O5 (α=.86) | Imagination and Intellect; avoids political O6 |
| TF | A3 (α=.77) + A6 (α=.75) | Altruism and Sympathy; best warmth/empathy content |
| JP | C2 (α=.82) + C5 (α=.85) | Orderliness and Self-Discipline; strongest JP signal |

That gives you 80 items (8 facets × 10 items) total — sufficient for IRT calibration. For an adaptive quiz you'd want to estimate item discrimination parameters (a-parameters in a 2PL model) by fitting to an existing dataset before going live.

**Recommended calibration dataset:** The Open Psychometrics IPIP-NEO data (n ≈ 1 million responses) is available at:  
https://openpsychometrics.org/_rawdata/  
(Look for the "IPIP-NEO data" download.)

---

## 7. Response Format and Administration

**Recommended response options (5-point Likert):**
1. Very Inaccurate
2. Moderately Inaccurate
3. Neither Accurate Nor Inaccurate
4. Moderately Accurate
5. Very Accurate

The IPIP has also been used with 4- and 7-point scales with no substantial loss of reliability. Binary true/false works but reduces score variance.

**Item ordering:** Distribute items from different scales/domains throughout the questionnaire, and alternate between +keyed and −keyed items. This discourages acquiescence bias (the tendency to agree with everything) and reduces the chance respondents realise which items measure the same construct.

**Instructions template (from Goldberg/Johnson):**
> "Describe yourself as you generally are now, not as you wish to be in the future. Describe yourself as you honestly see yourself, in relation to other people you know of the same sex as you are, and roughly your same age."

---

## 8. Licensing

All IPIP items are **public domain**. No permission is required. You may use, modify, translate, or commercialise them freely. The only request from the IPIP maintainers is that published research using IPIP scales cite the appropriate sources (listed per instrument above).

There is **no official IPIP test format** — the scoring keys on ipip.ori.org are the canonical artefacts, not the questionnaire layouts.

---

## 9. Key References

**Foundational:**
- Goldberg, L.R. (1992). The development of markers for the Big-Five factor structure. *Psychological Assessment, 4*, 26–42.
- Goldberg, L.R. (1999). A broad-bandwidth, public domain, personality inventory measuring the lower-level facets of several five-factor models. In Mervielde et al. (Eds.), *Personality psychology in Europe* (Vol. 7, pp. 7–28).
- Goldberg, L.R., et al. (2006). The international personality item pool and the future of public-domain personality measures. *Journal of Research in Personality, 40*, 84–96.

**IPIP-NEO-120:**
- Johnson, J.A. (2014). Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory. *Journal of Research in Personality, 51*, 78–89.

**Big Five ↔ MBTI mapping:**
- McCrae, R.R., & Costa, P.T. (1989). Reinterpreting the Myers-Briggs Type Indicator from the perspective of the five-factor model of personality. *Journal of Personality, 57*, 17–40.  
  *(This is the foundational paper establishing the E=EI, O=NS, A=TF, C=JP mapping.)*

**IRT / Adaptive testing:**
- Maples-Keller, J.L., et al. (2019). Using item response theory to develop a 60-item representation of the NEO PI-R using the IPIP. *Psychological Assessment, 31*, 986–997.

---

## 10. URLs Quick Reference

| Resource | URL |
|----------|-----|
| IPIP home | https://ipip.ori.org |
| 50-item Big Five key | https://ipip.ori.org/newBigFive5broadKey.htm |
| 50-item questionnaire layout | https://ipip.ori.org/New_IPIP-50-item-scale.htm |
| 300-item NEO facets key | https://ipip.ori.org/newNEOFacetsKey.htm |
| 120-item NEO key | https://ipip.ori.org/newNEOKey.htm |
| All scales by construct | https://ipip.ori.org/newIndexofScaleLabels.htm |
| Item assignment table | https://ipip.ori.org/ItemAssignmentTable.htm |
| Scoring instructions | https://ipip.ori.org/newScoringInstructions.htm |
| Raw dataset (calibration) | https://openpsychometrics.org/_rawdata/ |
| Online IPIP-NEO (live test) | https://ipip.ori.org/newNEOKey.htm |
| Johnson's IPIP-NEO site | https://www.personal.psu.edu/j5j/IPIP/ |
