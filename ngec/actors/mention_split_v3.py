"""
split_mention_v3 -- take an actor mention apart into (core_query, actor_desc, country) WITHOUT a role lexicon.

Signals, all of which a deployment can customise by swapping a data file rather than editing code:
  1. the PLOVER agents file  -- "is this prefix a job/role description?"  = max cosine between the prefix
                                (country and modifiers stripped) and the agent patterns, under the fast
                                static embedding model. New agent patterns automatically become role words.
  2. the Wikipedia index     -- "is this suffix a name?" = exact title / redirect / alternative-name match
                                (one cheap `terms` query on the .keyword fields for all candidate spans).
                                Optional: without an ES client the capitalisation shape stands in for it.
  3. capitalisation shape    -- a right-anchored run of capitalised tokens (name particles and inner
                                function words allowed) is name-like.
  4. CountryDetector         -- home country vs target country ("ambassador TO China").
The only word lists are closed-class: determiners, prepositions, and a dozen role modifiers ("former",
"acting", ...) that are treated as description but never decide a split on their own.

The mention is scanned at every split point j: prefix = tokens[:j] (description), suffix = tokens[j:]
(core). Apposition ("Robin Brooks, a senior fellow at ...") is handled first by scoring comma segments.
Returns the same keys as the v1 splitter so it can be dropped in.

Status: NOT the pipeline default. `actor_resolution.SPLITTER` is still "v1";
pass `splitter="v3"` (or set that constant) to use this one. It came out of the
NESS/Wikidata linking study of 2026-09-18/19 as the lexicon-free answer to the
rejected v2 splitter, and it is committed here so the work is under version
control and can be evaluated in place.

Measured when it was frozen (2026-09-19), each with the shipped ranker:

    | evaluation                        |    v1 |    v2 |    v3 |
    |-----------------------------------|-------|-------|-------|
    | VOA gold end-to-end (1,823 spans) | 0.919 | 0.929 | 0.925 |
    | long noun phrases (600)           | 0.615 | 0.518 | 0.660 |
    | repo case table (core query)      | 0.765 | 1.000 | 0.804 |

v2 wins its own case table because that table was written to describe it. The
long-noun-phrase column is the one this rewrite exists for.

Costs that the v1 splitter does not have: a sentence-transformer encoder (the
agent patterns are embedded once at construction) and one Elasticsearch
`terms` query per mention. Build it once and keep it -- see
`actor_resolution._v3_splitter`.
"""
import re, unicodedata
from functools import lru_cache
import numpy as np, requests

DETS = {"the", "a", "an", "this", "that", "these", "those", "its", "his", "her", "their", "our"}
PREPS = {"of", "for", "to", "in", "at", "on", "from", "with", "under", "by"}
MODIFIERS = {"former", "ex", "ex-", "acting", "interim", "outgoing", "incoming", "then", "late", "current", "new",
             "senior", "deputy", "vice", "chief", "top", "veteran", "longtime", "controversial", "embattled", "so-called"}
PARTICLES = {"de", "da", "del", "della", "di", "du", "der", "van", "von", "bin", "ibn", "al", "el", "la", "le", "y", "e", "dos", "das", "st.", "san"}
CONNECTORS = {"of", "for", "and", "the", "&", "de", "la", "du", "des", "del", "al", "on", "in"}
STATIC_ENCODER = "sentence-transformers/static-retrieval-mrl-en-v1"
# Infobox template families of things that act: people and organisations of every kind. A page whose
# infobox is not one of these ("Force" the physical quantity, "Corruption" the film) is not an actor name
# for splitting purposes. This is Wikipedia's own template taxonomy, not a user-facing word list.
PERSON_BOX = re.compile(r"person|biography|officeholder|royalty|noble|monarch|politician|judge|clergy|saint", re.I)
ACTOR_BOX = re.compile(r"person|biography|officeholder|royalty|noble|monarch|leader|politician|judge|clergy|saint|"
                       r"organi[sz]ation|party|agency|company|corporation|military|war faction|militant|rebel|union|"
                       r"university|school|institute|college|club|team|newspaper|broadcast|network|website|church|"
                       r"religious|government|ministry|department|legislature|court|police|law enforcement|"
                       r"central bank|bank|ngo|non-profit|nonprofit|think tank|association|federation|council|committee|"
                       r"country|state|territory|building|palace|residence|cabinet", re.I)


def load_agent_patterns(agents_file):
    """The PLOVER agents file, cleaned the way AgentMatcher does it (patterns only, codes dropped)."""
    data = open(agents_file, encoding="utf-8").read()
    data = re.sub(r"\{.+?\}", "", data)
    out = []
    for line in data.split("\n"):
        if not line or line.startswith(("#", "!")):
            continue
        line = re.sub(r"#.+", "", line).strip()
        if not re.search(r"\[.+?\]", line):
            continue
        pat = re.sub(r"\[.+?\]", "", line).replace("_", " ").lower().strip()
        if "!minist!" in pat:
            out += [pat.replace("!minist!", r) for r in ("minister", "ministry")]
        elif "!person!" in pat:
            out += [pat.replace("!person!", r) for r in ("person", "man", "woman")]
        elif pat:
            out.append(pat)
    return sorted(set(out))


class SplitterV3:
    def __init__(self, country_detector, agents_file, es_url="http://localhost:9200/wiki", encoder=None, nlp=None):
        from sentence_transformers import SentenceTransformer
        self.cd = country_detector
        # The splitter finds countries by name (it needs the name to tell "Bank
        # of Uganda" from "Ugandan bank"), but the `country` it returns is an
        # ISO-3 code, as v1's is: the agent matcher and code selector expect one.
        self.name_to_iso3 = dict(zip(country_detector.countries["Name"], country_detector.countries["CCA3"]))
        self.es_url = es_url
        self.nlp = nlp
        self.enc = encoder or SentenceTransformer(STATIC_ENCODER, device="cpu")
        self.patterns = load_agent_patterns(agents_file)
        P = self.enc.encode(self.patterns, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        self.P = np.asarray(P)

    # ---------------------------------------------------------------- signals
    @lru_cache(maxsize=100000)
    def role_score(self, text):
        """max cosine between `text` and any agents-file pattern (0 if empty)."""
        text = text.strip(" ,;:")
        if not text:
            return 0.0
        v = self.enc.encode([text.lower()], normalize_embeddings=True, show_progress_bar=False)[0]
        return float((self.P @ v).max())

    def wiki_names(self, spans):
        """subset of `spans` that is an exact article title / redirect / alternative name (any of 3 casings)."""
        if not self.es_url or not spans:
            return set()
        variants = {}
        for s in spans:
            forms = [s, s[:1].upper() + s[1:]] + ([s.title()] if s.isupper() else [])
            for v in forms:
                variants.setdefault(v, s)
        q = {"size": 500, "_source": ["title", "redirects", "alternative_names", "box_type"],
             "query": {"bool": {"should": [{"terms": {"title.keyword": list(variants)}},
                                           {"terms": {"redirects.keyword": list(variants)}},
                                           {"terms": {"alternative_names.keyword": list(variants)}}]}}}
        try:
            hits = requests.post(f"{self.es_url}/_search", json=q, timeout=5).json()["hits"]["hits"]
        except Exception:
            return set()
        found = {}
        for h in hits:
            s = h["_source"]
            names = {s["title"]} | set(s.get("redirects") or []) | set(s.get("alternative_names") or [])
            bt = s.get("box_type") or ""
            entity = ("person" if PERSON_BOX.search(bt) else "org") if ACTOR_BOX.search(bt) else False
            for v, orig in variants.items():
                if v in names:
                    found[orig] = found.get(orig) or entity
        return found

    @staticmethod
    def cap_shape(tokens):
        """1.0 if every token is capitalised (particles / inner connectors allowed), else the capitalised share."""
        if not tokens:
            return 0.0
        core = [t for t in tokens]
        ok = 0
        for i, t in enumerate(core):
            w = t.strip("\"'(),.;:")
            if not w:
                continue
            if w[0].isupper() or re.match(r"^(al|el|d|l|bin|ibn|abu|ben|de|van|von)[-']", w, re.I):
                ok += 1
            elif w.lower() in PARTICLES or (0 < i < len(core) - 1 and w.lower() in CONNECTORS):
                ok += 1
        return ok / len(core)

    # ---------------------------------------------------------------- split
    def split(self, text, context="", known_country=""):
        raw = re.sub(r"[’']s$", "", text.strip())
        alt_terms = []
        # --- parenthetical acronym / qualifier: "Kabaka Yekka (KY)", "(EPRP) The ...", "Los Rastrojos (Colombia)"
        paren_country = ""
        def _paren(m_p):
            nonlocal paren_country
            inner = m_p.group(1).strip()
            c_in, rest_in = self.cd.search_nat(inner.title() if inner.isupper() else inner, use_name=True)
            if c_in and not rest_in.strip():                                 # "(Colombia)", "(COLOMBIA)": the country qualifier
                paren_country = c_in
                return " "
            if re.fullmatch(r"[A-Z0-9][A-Z0-9.&/-]{1,9}", inner):            # an acronym: drop it from the scan (kept in the pass-through core)
                return " "
            return m_p.group(0)
        raw = re.sub(r"\s*\(([^()]{1,40})\)\s*", _paren, raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        # --- country: in the span (home unless it follows to/in/for after a role), else context, else none
        target_country, m = "", None
        for mm in re.finditer(r"\b(to|in|for)\s+(the\s+)?([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})", raw):
            words = mm.group(3).split()
            for k in range(len(words), 0, -1):                 # longest capitalised phrase that IS a country
                phrase = " ".join(words[:k])
                c, rest = self.cd.search_nat(phrase, use_name=True)
                if c and not rest.strip():
                    target_country, m = c, re.match(r".*", mm.group(1) + " " + (mm.group(2) or "") + phrase)
                    m_span = (mm.start(), mm.start() + len(mm.group(1) + " " + (mm.group(2) or "") + phrase))
                    break
            if target_country:
                break
        span_wo_target = (raw[: m_span[0]] + " " + raw[m_span[1]:]).strip() if target_country else raw
        target_phrase = raw[m_span[0]: m_span[1]] if target_country else ""
        code, trimmed = self.cd.search_nat(span_wo_target, use_name=True)
        home_country = code or ""
        if code:
            # Only an adjectival or possessive country reference is a modifier to cut ("Egyptian ...", "Mexico's ...").
            # A bare country NAME inside the mention is part of the name itself ("Bank of Uganda",
            # "Sudan Liberation Movement", "Party of Democratic Action of Croatia") and stays in the query.
            name_pat = r"(?<![\w-])" + re.escape(code) + r"(?![\w-])(?!['’]s)"
            bare_name_present = bool(re.search(name_pat, span_wo_target, flags=re.I if span_wo_target.isupper() else 0))
            if bare_name_present:
                trimmed = span_wo_target
            elif re.sub(r"^(the|a|an)(\s+|$)", "", trimmed, flags=re.I).strip() and trimmed != span_wo_target:
                # a demonym inside an organisation's own name ("Ethiopian People's Revolutionary Party",
                # "Nigerian Army"): keep it when an article is titled exactly that
                bare_full = re.sub(r"^(the|a|an)\s+", "", span_wo_target, flags=re.I)
                if self.wiki_names([bare_full]).get(bare_full):
                    trimmed = span_wo_target
        home_country = home_country or paren_country
        if target_country and not home_country:
            trimmed = span_wo_target
        elif target_country:
            trimmed = trimmed
        # v1 semantics: the description keeps "ambassador to China"; the country used for search is home
        if target_country:
            trimmed = f"{trimmed} {target_phrase}".strip()
        country_name = known_country or home_country or paren_country
        if not country_name and context:
            from ngec.actors.actor_resolution import country_from_context
            country_name = country_from_context(self.cd, text, context)

        self._alt = alt_terms
        # --- apposition: "Name, role ..." / "role ..., Name" / "Writer, TV host, and travel guide Name"
        segs = [x.strip() for x in re.split(r",\s+", trimmed) if x.strip()]
        if len(segs) >= 2:
            scored = []
            for i, seg in enumerate(segs):
                c, d = self._scan(seg)
                ctoks = c.split()
                bare_seg = re.sub(r"^(the|a|an)\s+", "", seg, flags=re.I)
                wiki = bool(self.wiki_names([c]).get(c))
                whole = int(bare_seg == c)                                   # the segment IS the name, no description around it
                shaped = int(self.cap_shape(ctoks) >= 0.99 and len(ctoks) >= 2)
                titleish = int(self.role_score(c) >= 0.9)                       # "Prime Minister", "Writer": a role, not a head
                scored.append((wiki + whole + shaped - titleish, -i, c, d))
            scored.sort(reverse=True)
            if scored[0][0] >= 2:
                _, negi, core, inner = scored[0]; i = -negi
                desc = ", ".join(x for k, x in enumerate(segs) if k != i)
                if inner:
                    desc = f"{inner}, {desc}" if desc else inner
                return self._result(text, raw, trimmed, core, desc, home_country, country_name, target_country, "apposition")
        scan_span = trimmed[: -len(target_phrase)].strip() if target_country else trimmed
        scan_span = re.sub(r"\s+(of|for|to|in|from|at)$", "", scan_span.strip(" ,;:'"))   # "King Abdullah II of" after the country was cut out
        if not re.sub(r"^(the|a|an)(\s+|$)", "", scan_span, flags=re.I).strip():
            bare = re.sub(r"^(the|a|an)(\s+|$)", "", raw, flags=re.I).strip()
            # a lone demonym / country word ("Israeli") is a country, not a searchable name; an organisation
            # that is also a country pattern ("the U.N.", "EU") stays a query
            core = "" if (len(bare.split()) == 1 and bare.isalpha()) else raw
            return self._result(text, raw, trimmed, core, "", home_country, country_name, target_country, "country-only")
        trail = re.search(r"\s+(of|from)\s+((?:the\s+)?[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})$", scan_span)
        core, desc = self._scan(scan_span)
        whole_known = re.sub(r"^(the|a|an)\s+", "", scan_span, flags=re.I) in self.wiki_names([re.sub(r"^(the|a|an)\s+", "", scan_span, flags=re.I)])
        if trail and not desc and not whole_known:
            head = scan_span[: trail.start()].strip()
            c2, d2 = self._scan(head)
            place_is_country = bool(self.cd.search_nat(trail.group(2), use_name=True)[0])
            head_kind = self.wiki_names([c2]).get(c2)
            head_known = head_kind == "person"                 # an organisation "of <Country>" keeps the country in its name
            person_shaped = not any(t.lower() in PREPS | CONNECTORS for t in c2.split())
            if self.cap_shape(c2.split()) >= 0.99 and len(c2.split()) >= 2 and self.role_score(c2) < 0.9 and person_shaped \
                    and head_kind != "org" and (place_is_country or head_known):     # "Pat Roberts of Kansas", not "Workers' Communist Party of Iraq"
                core, desc = c2, (d2 + " " + trail.group(0).strip()).strip()
        if target_country:
            if desc:
                desc = f"{desc} {target_phrase}".strip()
            else:                                            # "ambassador to China": generic, keep the target in the query
                core = f"{core} {target_phrase}".strip()
        return self._result(text, raw, trimmed, core, desc, home_country, country_name, target_country, "scan")

    ROLE_MIN = 0.45          # agents-file similarity above which a prefix counts as a description

    def _scan(self, span):
        """Best split of a comma-free span into (name suffix, description prefix).

        1. If some right-anchored suffix is a Wikipedia name (title/redirect/alt name) and looks like a
           name (capitalised, or several tokens), take the LONGEST such suffix whose prefix is either
           empty or descriptive (role-like under the agents file, or not fully capitalised).
        2. Otherwise the longest fully-capitalised right-anchored run whose prefix is role-like.
        3. Otherwise the whole span is the core (generic mention, or an unknown name with no title).
        """
        toks = span.split()
        while toks and toks[0].lower() in DETS:
            toks = toks[1:]
        if not toks:
            return span, ""
        n = len(toks)
        if " ".join(toks).isupper():
            return span, ""                                 # ALL CAPS ("NATIONAL LIBERATION ARMY"): shape says nothing; keep it whole
        cands = [" ".join(toks[j:]) for j in range(n)] + [" ".join(toks[:j]) for j in range(1, n)]
        names = self.wiki_names(cands)          # span -> True if the article has an infobox (an entity, not a concept)

        def descriptive_tok(t):
            # "Tripoli-based", "Iran-backed": hyphenated participles; "Mr.", "Dr.", "Gen.": abbreviations
            return t.lower() in MODIFIERS or bool(re.search(r"\w-\w+ed$", t)) or bool(re.fullmatch(r"[A-Z][a-z]{0,3}\.", t))

        def prefix_info(j):
            prefix = toks[:j]
            content = [t for t in prefix if t.lower() not in MODIFIERS | DETS | PREPS]
            role = self.role_score(" ".join(content)) if content else (0.6 if prefix else 0.0)
            pcap = self.cap_shape([t for t in content if not descriptive_tok(t)]) if content else 0.0
            if content and all(descriptive_tok(t) for t in content):
                role = max(role, 0.9)
            return role, pcap

        role_whole = self.role_score(" ".join(t for t in toks if t.lower() not in MODIFIERS | DETS))
        # the longest known name that starts at the first token ("Workers' Communist Party" in
        # "Workers' Communist Party of Iraq"): a cut inside it would tear an organisation's name
        left_end = max((k for k in range(2, n) if names.get(" ".join(toks[:k])) == "org"), default=0)
        # 1. wiki-name suffixes, longest first
        for j in range(n):
            suffix = toks[j:]
            if " ".join(suffix) not in names:
                continue
            if len(suffix) <= 2 and not names[" ".join(suffix)]:
                continue                                    # "boys", "State force", "Judiciary": a page, but not an actor's
            if len(suffix) == 1 and not suffix[0][:1].isupper():
                continue
            if suffix[0].lower() in PREPS | DETS or self.cap_shape(suffix) < 0.5:
                continue
            if j == 0:
                return " ".join(toks), ""
            role, pcap = prefix_info(j)
            if j == 1 and pcap >= 0.99 and (names[" ".join(suffix)] == "org" or j < left_end) and not descriptive_tok(toks[0]):
                continue                                    # "Workers' | Communist Party of Iraq", "Irish | Republican Army": part of the name
            if len(suffix) == 1 and (role < 0.9 or role < role_whole + 0.05):
                continue                                    # a lone capitalised word is a name only after a clear title ("Prime Minister Modi"),
                                                            # and not when the whole mention is itself an agent pattern ("Border Patrol")
            if role >= self.ROLE_MIN or pcap < 0.99:
                if toks[j - 1].lower() in PREPS | {"and", "&", "or", "-", "–", "—", "/"} and (role < 0.9 or len(suffix) == 1):
                    continue                                # don't cut right after "of"/"and" unless the prefix is clearly a title
                return " ".join(suffix), " ".join(toks[:j])
        # 1a. "<Name> of the <Org>": a multi-token entity name at the front, before an of/at/for phrase, is the head
        m_of = re.search(r"\s(of|at|for|with)\s", " ".join(toks))
        if m_of:
            left = " ".join(toks)[: m_of.start()].split()
            for j in range(len(left)):
                cand = " ".join(left[j:])
                if len(left) - j >= 2 and cand in names and names[cand] and self.cap_shape(left[j:]) >= 0.99:
                    role, pcap = prefix_info(j)
                    rest = " ".join(toks)[m_of.start():].strip()
                    rest_bare = re.sub(r"^(of|at|for|with)\s+(the\s+)?", "", rest)
                    c_rest, r_rest = self.cd.search_nat(rest_bare, use_name=True)
                    rest_known = bool(self.wiki_names([rest_bare]).get(rest_bare)) and not (c_rest and not r_rest.strip())   # "... of the French Institute ..." yes; "... of Iraq" no
                    if (j == 0 or role >= self.ROLE_MIN or pcap < 0.99) and rest_known:
                        return cand, (" ".join(left[:j]) + " " + rest).strip()
                    break
        # 1b. name first, generic role after it: "Boko Haram militants", "Islamic State group"
        for j in range(n - 1, 0, -1):
            prefix, suffix = toks[:j], toks[j:]
            p_txt = " ".join(prefix)
            if p_txt in names and names[p_txt] and self.cap_shape(prefix) >= 0.99 and all(t.islower() for t in suffix):
                role = self.role_score(" ".join(suffix))
                if (len(prefix) >= 2 and role >= self.ROLE_MIN) or (len(suffix) == 1 and role >= 0.9):
                    return p_txt, " ".join(suffix)
        # 2. capitalised run with a role-like prefix. The prefix must be MORE like an agent pattern than the
        #    whole mention is: "Prime Minister | Modi" yes; "Border Patrol", "Intelligence Agency" no -- the
        #    whole thing is the agent description, there is no name to cut off.
        for j in range(1, n):
            suffix = toks[j:]
            if self.cap_shape(suffix) < 0.99 or suffix[0].lower() in PREPS | DETS | PARTICLES:
                continue
            role, pcap = prefix_info(j)
            if role < role_whole + 0.05:
                continue
            if j == 1 and pcap >= 0.99 and (names.get(" ".join(suffix)) == "org" or j < left_end) and not descriptive_tok(toks[0]):
                continue                                    # "Islamic | Movement", "Workers' | Communist Party": part of the organisation's name
            if (role >= 0.9 or (role >= self.ROLE_MIN and pcap < 0.99)) and toks[j - 1].lower() not in PREPS | {"and", "&", "or", "-", "–", "—", "/"}:
                return " ".join(suffix), " ".join(toks[:j])
        # 3. degenerate: keep the span as it came
        return span, ""

    def _result(self, text, raw, trimmed, core, desc, home_country, country_name, target_country, path):
        core = re.sub(r"[’']s$", "", core.strip(" ,;:"))
        if not desc and path in ("scan", "apposition") and core:
            # No role description was found, so the mention IS a name: pass it through as written (demonym,
            # parenthetical acronym or country qualifier included -- "Eritrean Salvation Front", "Kabaka Yekka (KY)").
            # The country and the acronym are still reported separately for retrieval.
            core = re.sub(r"[’']s$", "", text.strip())
        # the description is the mention minus the core, so a nationality adjective stays in it
        # ("Ethiopian Prime Minister"): evidence for the ranker, while the country is reported separately
        if desc and core and core in raw:
            desc = re.sub(r"\s+", " ", raw.replace(core, " ", 1)).strip(" ,;:")
            desc = re.sub(r"^(the|a|an)\s+", "", desc, flags=re.I)
        desc = re.sub(r"\s+(of|for|to|at|in|the|a|an)$", "", desc.strip(" ,;:"), flags=re.I)
        if desc and all(re.fullmatch(r"[A-Z][a-z]{0,3}\.", t) for t in desc.split()):
            desc = ""
        return {"text": text, "country": self.name_to_iso3.get(home_country, home_country) or None, "country_name": country_name or "", "target_country": target_country,
                "trimmed_text": trimmed, "core_query": core, "actor_desc": desc,
                "ner_extracted_specific": bool(desc), "alt_query_terms": ([raw, trimmed] if desc else []) + getattr(self, "_alt", []),
                "ents": [], "ent_text": core, "doc": None, "split_path": path}
