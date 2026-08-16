"""
Fast-path answering: pattern-matches common Padres-stat question shapes and
answers them with a direct SQL lookup, with zero Gemini calls. Falls through
(returns None) for anything that doesn't clearly match a known shape, so the
full LangChain SQL agent stays available as a fallback for everything else.

Covers:
  - notable single-game feats           ("how many no-hitters have they thrown", "cycles", "perfect games")
  - compound single-game feats          ("who's stolen 3 bases and hit a HR in one game")
  - compound season/career milestones   ("who's had 30 HR and 30 SB in a season")
  - hitting / pitcher win streaks       ("longest hitting streak", "most consecutive wins")
  - shutouts                            ("who has the most shutouts")
  - two-player comparisons              ("who had more HR, Machado or Tatis")
  - head-to-head vs an opponent         ("how have the Padres done against the Dodgers")
  - player stat in a specific season    ("how many HR did Tatis hit in 2021")
  - player career stat total            ("how many career saves does Hoffman have")
  - player stat trend across seasons    ("Tatis's home runs by year")
  - player rookie-season stat           ("how many HR did Tatis hit as a rookie")
  - player bio (position/tenure/bats)   ("what position did Gwynn play")
  - tenure/longevity leaders            ("who's played the most games for the Padres")
  - season standings                    ("what was the record in 2024")
  - best/worst season                   ("what was the Padres' best season")
  - postseason results for a year       ("who did they lose to in the playoffs in 2024")
  - postseason history overall          ("have the Padres ever won a World Series")
  - postseason-only leaderboards        ("most home runs in Padres postseason history")
  - team season totals                  ("how many home runs did the team hit in 2021")
  - season/career leader for a stat     ("who has the most home runs")
"""
import re
import sqlite3
import difflib

DB_PATH = "padres_history.db"

STOPWORDS = {
    "Padres", "San", "Diego", "Who", "What", "How", "When", "Where", "Did",
    "Does", "Has", "Had", "Was", "Is", "The", "In", "During", "World",
    "Series", "National", "League", "MLB",
}

# (regex, column, batting/pitching, sort direction, display label)
STAT_PATTERNS = [
    (r"home ?runs?|\bhrs?\b", "HR", "batting", "DESC", "home runs"),
    (r"\d+\s+hits?\b|\bhits\b", "H", "batting", "DESC", "hits"),  # not bare "hit" -- collides with the verb ("hit a home run")
    (r"\brbis?\b|runs? batted in", "RBI", "batting", "DESC", "RBIs"),
    (r"\bstolen bases?\b|\bsteals?\b|\bstole\b|\bstolen\b", "SB", "batting", "DESC", "stolen bases"),
    (r"\bdoubles?\b", "Doubles", "batting", "DESC", "doubles"),
    (r"\btriples?\b", "Triples", "batting", "DESC", "triples"),
    (r"batting averages?|\bavgs?\b", "AVG", "batting", "DESC", "batting average"),
    (r"\bwalks?\b|base on balls", "BB", "batting", "DESC", "walks"),
    (r"\bwins?\b", "W", "pitching", "DESC", "wins"),
    (r"\bsaves?\b", "SV", "pitching", "DESC", "saves"),
    (r"strikeouts?|\bk'?s\b", "SO", "pitching", "DESC", "strikeouts"),
    (r"\bera\b|earned run average", "ERA", "pitching", "ASC", "ERA"),
]

SUPERLATIVE_RE = re.compile(
    r"\bmost\b|\bleader\b|\bleads?\b|\bled\b|\ball[- ]time\b|\bcareer\b|\btotal\b|"
    r"\brecord\b|\bhighest\b|\blowest\b|\bbest\b|\bsingle[- ]season\b"
)

TEAM_ALIASES = {
    "angels": ["ANA"], "diamondbacks": ["ARI"], "d-backs": ["ARI"], "dbacks": ["ARI"],
    "athletics": ["ATH", "OAK"], "a's": ["ATH", "OAK"], "braves": ["ATL"],
    "orioles": ["BAL"], "red sox": ["BOS"], "white sox": ["CHA"], "cubs": ["CHN"],
    "reds": ["CIN"], "guardians": ["CLE"], "indians": ["CLE"], "rockies": ["COL"],
    "tigers": ["DET"], "marlins": ["FLO", "MIA"], "astros": ["HOU"], "royals": ["KCA"],
    "dodgers": ["LAN"], "brewers": ["MIL"], "twins": ["MIN"], "expos": ["MON"],
    "yankees": ["NYA"], "mets": ["NYN"], "phillies": ["PHI"], "pirates": ["PIT"],
    "mariners": ["SEA"], "giants": ["SFN"], "cardinals": ["SLN"], "rays": ["TBA"],
    "rangers": ["TEX"], "blue jays": ["TOR"], "nationals": ["WAS"],
}

TAG = " *(instant answer from local database)*"


def _conn():
    return sqlite3.connect(DB_PATH)


def _bare_name(name):
    return re.sub(r"\s*\(\d{4}(-\d{4})?\)\s*$", "", name).strip()


def find_player(conn, fragment):
    """Return the single matching padres_players.player_name, or None if no/ambiguous match."""
    fragment = fragment.strip().strip(".,?!'’")
    if not fragment:
        return None
    names = [r[0] for r in conn.execute("SELECT DISTINCT player_name FROM padres_players").fetchall()]
    frag_low = fragment.lower()

    exact = [n for n in names if _bare_name(n).lower() == frag_low]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None  # ambiguous (e.g. two "Dave Roberts") -- let the full agent sort it out

    contains = [n for n in names if frag_low in _bare_name(n).lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        return None

    bare_map = {_bare_name(n): n for n in names}
    close = difflib.get_close_matches(fragment, list(bare_map.keys()), n=2, cutoff=0.8)
    if len(close) == 1:
        return bare_map[close[0]]
    return None


def extract_name_candidates(question):
    spans = re.findall(r"\b[A-Z][a-zA-Z'.\-]+(?:\s+[A-Z][a-zA-Z'.\-]+){0,2}\b", question)
    spans = sorted(set(spans), key=len, reverse=True)
    return [s for s in spans if s.split()[0] not in STOPWORDS]


def find_all_players(conn, question):
    """Like find_player, but returns every distinct player resolved from the
    question (for comparison questions naming two players), in order found.
    """
    found = []
    seen = set()
    for cand in extract_name_candidates(question):
        p = find_player(conn, cand)
        if p and p not in seen:
            found.append(p)
            seen.add(p)
    return found


def find_year(question):
    m = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", question)
    return int(m.group(1)) if m else None


def find_stat(question):
    q = question.lower()
    for pattern, col, kind, sort, label in STAT_PATTERNS:
        if re.search(pattern, q):
            return col, kind, sort, label
    return None


def find_opponent(question):
    q_lower = question.lower()
    matches = [(name, codes) for name, codes in TEAM_ALIASES.items() if re.search(r"\b" + re.escape(name) + r"\b", q_lower)]
    if len(matches) == 1:
        return matches[0][1], matches[0][0]
    return None, None


NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUM_RE = re.compile(r"\b(\d+|" + "|".join(NUM_WORDS) + r")\b")


def find_all_stats_with_thresholds(question):
    """Find every stat mentioned in the question, each with its own numeric
    threshold (e.g. "stolen 3 bases" -> (SB, 3), "hit a home run" -> (HR, 1)
    since no number was given). Prefers a number that comes *before* the stat
    mention -- "100 wins" is the dominant English construction -- and only
    looks after it as a fallback (for verb-first phrasing like "stole 3
    bases"). Preferring "before" avoids misassignment when two stat phrases
    sit next to each other: in "100 career wins and 1000 career strikeouts",
    a plain nearest-in-either-direction search would wrongly give "wins" the
    1000 from the neighboring phrase, since it's fewer characters away than
    its own "100" -- searching backward first gets this right.
    """
    q = question.lower()
    results = []
    for pattern, col, kind, sort, label in STAT_PATTERNS:
        m = re.search(pattern, q)
        if not m:
            continue
        threshold = 1
        # the H pattern's own regex can embed a leading number in its match ("5 hits"
        # matches as one unit, via the "\d+\s+hits?\b" alternative) -- if so, that
        # number belongs to THIS stat; use it directly rather than searching around
        # the match, which would search past it and find a neighboring stat's number.
        embedded = re.match(r"(\d+)", m.group())
        if embedded:
            results.append((col, kind, int(embedded.group(1)), label))
            continue
        before = q[max(0, m.start() - 15):m.start()]
        best = None
        for num_m in NUM_RE.finditer(before):
            dist = m.start() - (max(0, m.start() - 15) + num_m.start())
            if best is None or dist < best[0]:
                best = (dist, num_m.group(1))
        if best is None:
            after = q[m.end():min(len(q), m.end() + 15)]
            for num_m in NUM_RE.finditer(after):
                dist = num_m.start()
                if best is None or dist < best[0]:
                    best = (dist, num_m.group(1))
        if best is not None:
            raw = best[1]
            threshold = int(raw) if raw.isdigit() else NUM_WORDS[raw]
        results.append((col, kind, threshold, label))
    return results


def _longest_hit_streak(conn):
    """Gaps-and-islands: longest run of consecutive games (by row order) with H>0."""
    return conn.execute("""
        WITH numbered AS (
            SELECT player_name, Date,
                   CASE WHEN H > 0 THEN 1 ELSE 0 END AS hit_game,
                   ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY Date) AS overall_rn
            FROM padres_batting_logs
        ),
        hits_only AS (
            SELECT player_name, Date, overall_rn,
                   ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY Date) AS hit_rn
            FROM numbered WHERE hit_game = 1
        )
        SELECT player_name, COUNT(*) AS streak_len, MIN(Date), MAX(Date)
        FROM hits_only
        GROUP BY player_name, overall_rn - hit_rn
        ORDER BY streak_len DESC
        LIMIT 1
    """).fetchone()


def _longest_win_streak(conn, career=False):
    """Gaps-and-islands: longest run of consecutive decisions (W or L) that were wins.
    career=True treats a pitcher's whole career as one sequence; otherwise each
    season is scoped separately and the best season-streak overall is returned.
    """
    partition = "player_name" if career else "player_name, Season"
    season_select = "NULL" if career else "Season"
    season_group = "" if career else ", Season"
    query = f"""
        WITH decisions AS (
            SELECT player_name, {season_select} AS Season, Date, W
            FROM padres_pitching_logs WHERE W=1 OR L=1
        ),
        numbered AS (
            SELECT player_name, Season, Date, W,
                   ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY Date) AS overall_rn
            FROM decisions
        ),
        wins_only AS (
            SELECT player_name, Season, Date, overall_rn,
                   ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY Date) AS win_rn
            FROM numbered WHERE W = 1
        )
        SELECT player_name, Season, COUNT(*) AS streak_len, MIN(Date), MAX(Date)
        FROM wins_only
        GROUP BY player_name{season_group}, overall_rn - win_rn
        ORDER BY streak_len DESC
        LIMIT 1
    """
    return conn.execute(query).fetchone()


def try_fast_path(question):
    conn = _conn()
    try:
        q_lower = question.lower()
        year = find_year(question)
        stat = find_stat(question)

        all_players = find_all_players(conn, question)
        player = all_players[0] if all_players else None

        # -1. two-player comparison -- e.g. "who had more HR, Machado or Tatis"
        if len(all_players) == 2 and stat:
            col, kind, sort, label = stat
            p1, p2 = all_players
            if year:
                table = "padres_season_batting" if kind == "batting" else "padres_season_pitching"
                row1 = conn.execute(f"SELECT {col} FROM {table} WHERE player_name=? AND Season=?", (p1, year)).fetchone()
                row2 = conn.execute(f"SELECT {col} FROM {table} WHERE player_name=? AND Season=?", (p2, year)).fetchone()
                scope = f"in {year}"
            else:
                table = "padres_career_batting" if kind == "batting" else "padres_career_pitching"
                row1 = conn.execute(f"SELECT {col} FROM {table} WHERE player_name=?", (p1,)).fetchone()
                row2 = conn.execute(f"SELECT {col} FROM {table} WHERE player_name=?", (p2,)).fetchone()
                scope = "for their career"
            if row1 and row2:
                v1, v2 = row1[0], row2[0]
                if v1 == v2:
                    verdict = "tied"
                else:
                    better = p1 if (v1 > v2) == (sort == "DESC") else p2
                    verdict = f"{better} had the better {label}"
                return f"{p1}: {v1} vs {p2}: {v2} in {label} {scope} -- {verdict}.{TAG}"
            return None

        # 0. notable single-game feats: no-hitters, perfect games, cycles, multi-HR games
        if re.search(r"\bperfect games?\b", q_lower):
            # true perfect game test: no hits/walks/HBP allowed AND exactly 27 batters
            # faced in a 9+ inning game. BFP=27 alone isn't sufficient -- a batter can
            # reach base (hit, walk, HBP, error) and still leave BFP at 27 if that
            # baserunner is erased the same inning (caught stealing, DP, pickoff), as
            # verified against Andrew Cashner's 2013-09-16 start (H=1, BFP=27 -- not
            # perfect). H=0/BB=0/HBP=0 rules out the common cases; BFP=27 additionally
            # catches reaching on an error when it isn't erased the same inning.
            rows = conn.execute(
                "SELECT player_name, Date FROM padres_pitching_logs "
                "WHERE Outs>=27 AND BFP=27 AND H=0 AND BB=0 AND HBP=0 ORDER BY Date"
            ).fetchall()
            if not rows:
                return f"No Padres pitcher has thrown a perfect game.{TAG}"
            listing = "; ".join(f"{name} ({date})" for name, date in rows)
            return f"The Padres have had {len(rows)} perfect game(s): {listing}.{TAG}"

        if re.search(r"no-?hitters?", q_lower):
            rows = conn.execute(
                "SELECT player_name, Date, SO FROM padres_pitching_logs WHERE H=0 AND Outs>=27 ORDER BY Date"
            ).fetchall()
            if not rows:
                return f"The Padres have not had a pitcher throw a no-hitter.{TAG}"
            listing = "; ".join(f"{name} ({date}, {so} K)" for name, date, so in rows)
            return f"The Padres have had {len(rows)} no-hitter(s) in franchise history: {listing}.{TAG}"

        if re.search(r"\bcycles?\b", q_lower):
            rows = conn.execute(
                'SELECT player_name, Date FROM padres_batting_logs '
                'WHERE (H - "2B" - "3B" - HR) >= 1 AND "2B" >= 1 AND "3B" >= 1 AND HR >= 1 ORDER BY Date'
            ).fetchall()
            if not rows:
                return f"No Padres player has hit for the cycle.{TAG}"
            listing = "; ".join(f"{name} ({date})" for name, date in rows)
            return f"The Padres have had {len(rows)} player(s) hit for the cycle: {listing}.{TAG}"

        if re.search(r"\bshutouts?\b", q_lower):
            if player:
                row = conn.execute(
                    "SELECT COUNT(*) FROM padres_pitching_logs WHERE player_name=? AND Outs>=27 AND ER=0", (player,)
                ).fetchone()
                count = row[0] if row else 0
                if not count:
                    return f"{player} has not thrown a shutout for the Padres.{TAG}"
                return f"{player} has thrown {count} shutout(s) for the Padres.{TAG}"
            if SUPERLATIVE_RE.search(q_lower):
                row = conn.execute(
                    "SELECT player_name, COUNT(*) c FROM padres_pitching_logs WHERE Outs>=27 AND ER=0 "
                    "GROUP BY player_name ORDER BY c DESC LIMIT 1"
                ).fetchone()
                if row:
                    name, count = row
                    return f"The Padres career leader in shutouts is {name}, with {count}.{TAG}"
                return None
            total = conn.execute("SELECT COUNT(*) FROM padres_pitching_logs WHERE Outs>=27 AND ER=0").fetchone()[0]
            return f"The Padres have had {total} shutout(s) thrown in franchise history.{TAG}"

        if re.search(r"\bthree[- ]?(home run|homer)|\b3[- ]?(home run|hr|homer)", q_lower) and re.search(r"games?", q_lower):
            rows = conn.execute(
                "SELECT player_name, Date, HR FROM padres_batting_logs WHERE HR>=3 ORDER BY Date"
            ).fetchall()
            if not rows:
                return f"No Padres player has hit 3+ home runs in a single game.{TAG}"
            listing = "; ".join(f"{name} ({date}, {hr} HR)" for name, date, hr in rows)
            return f"The Padres have had {len(rows)} game(s) with a player hitting 3+ home runs: {listing}.{TAG}"

        # 0b. compound single-game feat: two+ stats in the same game (e.g. "stole 3
        # bases and hit a HR in one game"). Requires explicit single-game framing so
        # it doesn't collide with career/season leader questions mentioning 2 stats.
        if re.search(r"\bsame game\b|\bsingle game\b|\bone game\b|\bin a game\b", q_lower):
            multi = find_all_stats_with_thresholds(question)
            kinds = {k for _, k, _, _ in multi}
            if len(multi) >= 2 and len(kinds) == 1:
                kind = kinds.pop()
                table = "padres_batting_logs" if kind == "batting" else "padres_pitching_logs"
                conditions = " AND ".join(f"{col} >= {threshold}" for col, _, threshold, _ in multi)
                cols_select = ", ".join(dict.fromkeys(col for col, _, _, _ in multi))
                rows = conn.execute(
                    f"SELECT player_name, Date, {cols_select} FROM {table} WHERE {conditions} ORDER BY Date"
                ).fetchall()
                desc = " and ".join(f"{threshold}+ {label}" for _, _, threshold, label in multi)
                if not rows:
                    return f"No Padres player has recorded {desc} in a single game.{TAG}"
                listing = "; ".join(f"{r[0]} ({r[1]})" for r in rows)
                return f"Padres player(s) with {desc} in a single game: {listing}.{TAG}"
            return None

        # 0b2. compound season/career milestone: two+ stats together over a season or
        # career (e.g. "30 HR and 30 SB in a season"). Only commits to this branch (and
        # only returns from inside it) once 2+ stats with a single kind are confirmed --
        # otherwise falls through untouched, since "career"/"single season" are also used
        # by the single-stat rules below (e.g. rule 4's "career home runs" lookup).
        if not player:
            season_kw = re.search(r"\bin a season\b|\bsingle season\b", q_lower)
            career_kw = re.search(r"\bcareer\b", q_lower)
            if season_kw or career_kw:
                multi = find_all_stats_with_thresholds(question)
                kinds = {k for _, k, _, _ in multi}
                if len(multi) >= 2 and len(kinds) == 1:
                    kind = kinds.pop()
                    season_scoped = bool(season_kw) and not career_kw
                    if season_scoped:
                        table = "padres_season_batting" if kind == "batting" else "padres_season_pitching"
                    else:
                        table = "padres_career_batting" if kind == "batting" else "padres_career_pitching"
                    conditions = " AND ".join(f"{col} >= {threshold}" for col, _, threshold, _ in multi)
                    cols_select = ", ".join(dict.fromkeys(col for col, _, _, _ in multi))
                    scope = "in a single season" if season_scoped else "for their career"
                    if season_scoped:
                        rows = conn.execute(
                            f"SELECT player_name, Season, {cols_select} FROM {table} WHERE {conditions} ORDER BY Season"
                        ).fetchall()
                        listing = "; ".join(f"{r[0]} ({r[1]})" for r in rows)
                    else:
                        rows = conn.execute(f"SELECT player_name, {cols_select} FROM {table} WHERE {conditions}").fetchall()
                        listing = "; ".join(f"{r[0]}" for r in rows)
                    desc = " and ".join(f"{threshold}+ {label}" for _, _, threshold, label in multi)
                    if not rows:
                        return f"No Padres player has recorded {desc} {scope}.{TAG}"
                    return f"Padres player(s) with {desc} {scope}: {listing}.{TAG}"

        # 0c. hitting streak
        if re.search(r"\bhit(?:ting)? streak\b|\bconsecutive (games? with a )?hits?\b", q_lower):
            row = _longest_hit_streak(conn)
            if row:
                name, length, start, end = row
                return (
                    f"The longest hitting streak in Padres history (based on team batting logs) is "
                    f"{length} consecutive games by {name}, from {start} to {end}.{TAG}"
                )
            return None

        # 0d. pitcher win streak
        if re.search(r"\bconsecutive (pitcher )?wins?\b|\bwin(?:ning)? streak\b", q_lower):
            career = bool(re.search(r"\bcareer\b|\ball[- ]time\b|\bever\b", q_lower))
            row = _longest_win_streak(conn, career=career)
            if row:
                name, _season, length, start, end = row
                scope = "" if career else " in a single season"
                return f"The most consecutive wins by a Padres pitcher{scope} is {length}, by {name} ({start} to {end}).{TAG}"
            return None

        # 1. head-to-head vs a specific opponent, no stat involved
        if not stat:
            codes, opp_name = find_opponent(question)
            if codes and re.search(r"\bagainst\b|\bvs\.?\b|\bversus\b|\brecord (against|vs)\b|\bhow (have|has|do) the padres (do|done|fare)\b", q_lower):
                placeholders = ",".join("?" * len(codes))
                row = conn.execute(
                    f"SELECT SUM(Win), SUM(Loss) FROM padres_games WHERE Opponent IN ({placeholders}) AND GameType='regular'",
                    codes,
                ).fetchone()
                if row and row[0] is not None:
                    wins, losses = row
                    return f"The Padres are {wins}-{losses} all-time against the {opp_name.title()} (regular season).{TAG}"
                return None

        # 1b. tenure/longevity leaders -- "who's played the most games/seasons for the Padres"
        if not player and not stat and re.search(
            r"\bmost (games|seasons)\b|\blongest tenure\b|\bplayed the (most|longest)\b", q_lower
        ):
            if re.search(r"\bseasons?\b", q_lower) and not re.search(r"\bgames?\b", q_lower):
                row = conn.execute(
                    "SELECT player_name, seasons_with_padres FROM padres_players ORDER BY seasons_with_padres DESC LIMIT 1"
                ).fetchone()
                if row:
                    name, seasons = row
                    return f"{name} played the most seasons for the Padres, with {seasons}.{TAG}"
                return None
            row = conn.execute(
                "SELECT player_name, total_games FROM padres_players ORDER BY total_games DESC LIMIT 1"
            ).fetchone()
            if row:
                name, games = row
                return f"{name} played the most games for the Padres, with {games}.{TAG}"
            return None

        # 1c. postseason-only stat totals/leaders -- distinct from padres_career_* (which
        # combines regular + postseason, per the Gwynn hits discrepancy found earlier) and
        # from rule 9 below (which handles year+series game-by-game W/L, not a stat total).
        # Placed before rule 4's generic "career" stat lookup since a question like "career
        # postseason home runs" should scope to postseason games only, not everything.
        if stat and stat[0] not in ("AVG", "ERA") and re.search(r"\bpostseason\b|\bplayoffs?\b|\bworld series\b", q_lower):
            col, kind, sort, label = stat
            table = "padres_batting_logs" if kind == "batting" else "padres_pitching_logs"
            conditions = ["GameType != 'regular'"]
            params = []
            if player:
                conditions.append("player_name = ?")
                params.append(player)
            if year:
                conditions.append("Season = ?")
                params.append(year)
            where = " AND ".join(conditions)

            if player:
                row = conn.execute(f"SELECT SUM({col}) FROM {table} WHERE {where}", params).fetchone()
                if row and row[0] is not None:
                    scope = f" in {year}" if year else ""
                    return f"{player} had {row[0]} {label} in Padres postseason play{scope}.{TAG}"
                return None
            elif SUPERLATIVE_RE.search(q_lower):
                row = conn.execute(
                    f"SELECT player_name, SUM({col}) s FROM {table} WHERE {where} GROUP BY player_name ORDER BY s {sort} LIMIT 1",
                    params,
                ).fetchone()
                if row:
                    name, value = row
                    return f"The Padres postseason leader in {label} is {name}, with {value}.{TAG}"
                return None
            elif year:
                row = conn.execute(f"SELECT SUM({col}) FROM {table} WHERE {where}", params).fetchone()
                if row and row[0] is not None:
                    return f"The Padres had {row[0]} total {label} in their {year} postseason run.{TAG}"
                return None
            # else: no player, no superlative, no year -- too ambiguous, fall through

        # 2. player + year + stat -> season stat lookup
        if player and year and stat:
            col, kind, _, label = stat
            table = "padres_season_batting" if kind == "batting" else "padres_season_pitching"
            row = conn.execute(f"SELECT {col} FROM {table} WHERE player_name=? AND Season=?", (player, year)).fetchone()
            if row is not None:
                return f"{player} had {row[0]} {label} in {year}.{TAG}"
            return None

        # 3. player + stat trend across seasons, no specific year -- checked before the
        # career-total rule below since "total" alone (e.g. "home run total by season")
        # would otherwise false-match there.
        if player and stat and not year and re.search(r"by year|by season|each season|every season|over the years|season by season", q_lower):
            col, kind, _, label = stat
            table = "padres_season_batting" if kind == "batting" else "padres_season_pitching"
            rows = conn.execute(f"SELECT Season, {col} FROM {table} WHERE player_name=? ORDER BY Season", (player,)).fetchall()
            if rows:
                listing = "; ".join(f"{season}: {value}" for season, value in rows)
                return f"{player}'s {label} by season with the Padres -- {listing}.{TAG}"
            return None

        # 4. player + career/total + stat, no year -> career stat lookup
        if player and stat and not year and re.search(r"\bcareer\b|\btotal\b|\ball[- ]time\b", q_lower):
            col, kind, _, label = stat
            table = "padres_career_batting" if kind == "batting" else "padres_career_pitching"
            row = conn.execute(f"SELECT {col} FROM {table} WHERE player_name=?", (player,)).fetchone()
            if row is not None:
                return f"{player} had {row[0]} career {label} with the Padres.{TAG}"
            return None

        # 5. player + rookie/debut season + stat
        if player and stat and not year and re.search(r"\brookie\b|\bdebut\b|\bfirst season\b", q_lower):
            prow = conn.execute("SELECT first_season FROM padres_players WHERE player_name=?", (player,)).fetchone()
            if not prow:
                return None
            first_season = prow[0]
            col, kind, _, label = stat
            table = "padres_season_batting" if kind == "batting" else "padres_season_pitching"
            row = conn.execute(f"SELECT {col} FROM {table} WHERE player_name=? AND Season=?", (player, first_season)).fetchone()
            if row is not None:
                return f"In his rookie season ({first_season}), {player} had {row[0]} {label}.{TAG}"
            return None

        # 6. player bio -- position / tenure / bats-throws
        if player and not stat and re.search(
            r"\bposition\b|\bbats?\b|\bthrows?\b|\byears? (with|did|was)|\btenure\b|\bwhen (was|did)\b|\bdebut\b",
            q_lower,
        ):
            row = conn.execute(
                "SELECT primary_position, Bats, Throws, first_season, last_season FROM padres_players WHERE player_name=?",
                (player,),
            ).fetchone()
            if row:
                pos, bats, throws, first, last = row
                years = f"{first}" if first == last else f"{first}-{last}"
                return (
                    f"{player} primarily played {pos} for the Padres ({years}), "
                    f"batting {bats} and throwing {throws}.{TAG}"
                )
            return None

        # 7. season standings -- year + record/standings keywords, no player
        if year and not player and re.search(r"\brecord\b|\bstandings?\b|\bwin[- ]?loss\b|\bhow many (games|wins)\b", q_lower):
            row = conn.execute("SELECT Wins, Losses, WinPct FROM padres_season_standings WHERE Season=?", (year,)).fetchone()
            if row:
                wins, losses, pct = row
                return f"The Padres finished {year} with a record of {wins}-{losses} ({pct} winning percentage).{TAG}"
            return None

        # 8. best/worst season overall, no year/player/stat
        if not player and not stat and not year and re.search(r"\b(best|worst|greatest)\s+(season|record)\b", q_lower):
            order = "DESC" if re.search(r"\bbest\b|\bgreatest\b", q_lower) else "ASC"
            row = conn.execute(f"SELECT Season, Wins, Losses, WinPct FROM padres_season_standings ORDER BY WinPct {order} LIMIT 1").fetchone()
            if row:
                season, wins, losses, pct = row
                qualifier = "best" if order == "DESC" else "worst"
                return f"The Padres' {qualifier} season (by winning percentage) was {season}, finishing {wins}-{losses} ({pct}).{TAG}"
            return None

        # 9. postseason results for a specific year -- year + playoff keywords, no player
        if year and not player and re.search(r"\bplayoffs?\b|\bpostseason\b|\bworld series\b|\bnlds\b|\bnlcs\b|\bwild ?card\b", q_lower):
            rows = conn.execute(
                "SELECT Opponent, GameType, Win, Loss FROM padres_postseason_games WHERE Season=? ORDER BY Date", (year,)
            ).fetchall()
            if not rows:
                return f"The Padres did not make the postseason in {year}.{TAG}"
            series = {}
            for opp, gtype, win, loss in rows:
                w, l = series.get((gtype, opp), (0, 0))
                series[(gtype, opp)] = (w + win, l + loss)
            summary = "; ".join(f"{gtype} vs {opp}: {w}-{l}" for (gtype, opp), (w, l) in series.items())
            return f"In {year}, the Padres' postseason: {summary}.{TAG}"

        # 10. postseason history overall -- no year, no player
        if not player and not year and re.search(
            r"how many times.*(playoffs|postseason)|world series appearances?|ever (won|win).{0,20}world series|"
            r"won a world series|world series titles?|world series championships?",
            q_lower,
        ):
            if "world series" in q_lower:
                ws_seasons = conn.execute(
                    "SELECT Season, SUM(Win) w, SUM(Loss) l FROM padres_postseason_games WHERE GameType='worldseries' GROUP BY Season"
                ).fetchall()
                ws_wins = [s for s, w, l in ws_seasons if w > l]
                if ws_wins:
                    return f"Yes -- the Padres have won the World Series in: {', '.join(str(s) for s in ws_wins)}.{TAG}"
                if ws_seasons:
                    years = ", ".join(str(s) for s, w, l in ws_seasons)
                    return f"No, the Padres have never won a World Series. They've appeared in it in {years}, losing each time.{TAG}"
                return f"No, the Padres have never appeared in a World Series.{TAG}"
            total = conn.execute("SELECT COUNT(DISTINCT Season) FROM padres_postseason_games").fetchone()[0]
            return f"The Padres have made the postseason {total} times in franchise history.{TAG}"

        # 11. team season totals (not a single player's leaderboard) -- keyed on phrases
        # that mean "sum it up", not bare "the team" (which also matches "who LED THE TEAM
        # in X", a leader-lookup question that must fall through to rule 12 instead).
        if stat and year and not player and re.search(r"\bthe team (hit|score|scored|allow|allowed)\b|\bteam total\b|\bteam combine[ds]?\b|\bcombined\b|\baltogether\b|\bin total\b", q_lower):
            col, kind, _, label = stat
            table = "padres_season_batting" if kind == "batting" else "padres_season_pitching"
            row = conn.execute(f"SELECT SUM({col}) FROM {table} WHERE Season=?", (year,)).fetchone()
            if row and row[0] is not None:
                return f"The Padres had {row[0]} total {label} as a team in {year}.{TAG}"
            return None

        # 12. season leader for a stat (year given) or single-season record (no year)
        if stat and not player and (year or re.search(r"single[- ]season|in a season|season record", q_lower)) and SUPERLATIVE_RE.search(q_lower):
            col, kind, sort, label = stat
            table = "padres_season_batting" if kind == "batting" else "padres_season_pitching"
            where = ["IP > 20"] if kind == "pitching" else []
            if year:
                where.append(f"Season = {year}")
            where_clause = ("WHERE " + " AND ".join(where)) if where else ""
            row = conn.execute(f"SELECT player_name, Season, {col} FROM {table} {where_clause} ORDER BY {col} {sort} LIMIT 1").fetchone()
            if row:
                name, season, value = row
                if year:
                    return f"In {year}, the Padres leader in {label} was {name} with {value}.{TAG}"
                return f"The Padres single-season record for {label} is {value}, by {name} in {season}.{TAG}"
            return None

        # 13. career leader for a stat, no year, no player
        if stat and not player and not year and SUPERLATIVE_RE.search(q_lower):
            col, kind, sort, label = stat
            table = "padres_career_batting" if kind == "batting" else "padres_career_pitching"
            where_clause = "WHERE IP > 300" if kind == "pitching" else ""
            row = conn.execute(f"SELECT player_name, {col} FROM {table} {where_clause} ORDER BY {col} {sort} LIMIT 1").fetchone()
            if row:
                name, value = row
                return f"The Padres career leader in {label} is {name}, with {value}.{TAG}"
            return None

        return None
    finally:
        conn.close()
