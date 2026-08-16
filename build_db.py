"""
Builds padres_history.db from the Retrosheet-style game-log CSVs
(allplayers.csv, batting.csv, pitching.csv).

MLB San Diego Padres team code is SDN (1969-present). Note: SDO also
appears in this dataset but is NOT the Padres -- it collides with an
unrelated 1937 team (Negro League barnstorming club featuring Satchel
Paige, Cool Papa Bell, etc.), so it's deliberately excluded.

Each game log row has a `stattype` of value/official/upper/lower for
batting, and value/official/upper/lower for pitching -- only `value` is
the real box-score line, the rest are duplicates/derived rows.

Tables built:
  padres_batting_logs   game-by-game batter box scores
  padres_pitching_logs  game-by-game pitcher box scores
  padres_games          one row per Padres game: result, opponent, game type
  padres_players        one row per player: bio, tenure, primary position

Aggregate tables built (materialized, not views -- langchain's SQLDatabase
view_support flag crashes on SQLite in this SQLAlchemy version, and without
it views are invisible to the agent, so these are built as real tables so
the chat agent can answer aggregate questions in one query instead of
hand-rolling GROUP BY/JOIN logic every time):
  padres_season_standings   wins/losses/win_pct by season (regular season)
  padres_postseason_games   playoff/LDS/LCS/WS games only
  padres_season_batting     batting totals + AVG by player+season
  padres_season_pitching    pitching totals + ERA by player+season
  padres_career_batting     career batting totals + AVG by player
  padres_career_pitching    career pitching totals + ERA by player
"""
import sqlite3
import pandas as pd

PADRES_TEAMS = {"SDN"}
CHUNKSIZE = 200_000

POSITION_COLS = {
    "g_p": "P", "g_c": "C", "g_1b": "1B", "g_2b": "2B", "g_3b": "3B",
    "g_ss": "SS", "g_lf": "LF", "g_cf": "CF", "g_rf": "RF", "g_dh": "DH",
}


def load_player_names():
    """Maps Retrosheet id -> display name. Disambiguates same-name players
    who both appear in Padres data (e.g. Tony Gwynn Sr. 1982-2001 vs.
    Tony Gwynn Jr. 2009-2010, both stored under id-distinct rows) by
    appending each player's own Padres tenure -- otherwise their stats
    would silently merge under one name in any GROUP BY player_name query.
    """
    players = pd.read_csv(
        "allplayers.csv",
        usecols=["id", "last", "first", "team", "season"],
        dtype={"id": str, "last": str, "first": str, "team": str},
    )
    players["player_name"] = (players["first"].fillna("") + " " + players["last"].fillna("")).str.strip()
    names = dict(zip(players["id"], players["player_name"]))

    sdn = players[players["team"].isin(PADRES_TEAMS)]
    tenure = sdn.groupby("id")["season"].agg(["min", "max"])
    id_to_name = sdn.drop_duplicates("id").set_index("id")["player_name"]

    name_counts = id_to_name.value_counts()
    colliding_names = set(name_counts[name_counts > 1].index)

    for pid, name in id_to_name.items():
        if name in colliding_names:
            first, last = tenure.loc[pid, "min"], tenure.loc[pid, "max"]
            span = f"{first}" if first == last else f"{first}-{last}"
            names[pid] = f"{name} ({span})"

    return names


def fmt_date(yyyymmdd):
    s = str(int(yyyymmdd))
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def build_batting(conn, names):
    cols = ["gid", "id", "team", "stattype", "b_ab", "b_h", "b_d", "b_t", "b_hr",
            "b_rbi", "b_w", "b_k", "b_sb", "b_cs", "date", "opp", "vishome", "gametype"]
    rows = []
    for chunk in pd.read_csv("batting.csv", usecols=cols, chunksize=CHUNKSIZE, low_memory=False):
        sub = chunk[(chunk["team"].isin(PADRES_TEAMS)) & (chunk["stattype"] == "value")]
        if not sub.empty:
            rows.append(sub)
        print(f"  batting: scanned {chunk.shape[0]:,} rows, kept {sub.shape[0]}", end="\r")

    print()
    padres_b = pd.concat(rows, ignore_index=True)
    padres_b["player_name"] = padres_b["id"].map(names).fillna(padres_b["id"])
    padres_b["Date"] = padres_b["date"].apply(fmt_date)
    padres_b["Season"] = padres_b["Date"].str[:4].astype(int)
    padres_b["home"] = padres_b["vishome"] == 1

    out = padres_b.rename(columns={
        "b_ab": "AB", "b_h": "H", "b_d": "2B", "b_t": "3B", "b_hr": "HR",
        "b_rbi": "RBI", "b_w": "BB", "b_k": "SO", "b_sb": "SB", "b_cs": "CS",
        "opp": "Opponent", "gametype": "GameType",
    })[["player_name", "Date", "Season", "team", "Opponent", "home", "GameType",
        "AB", "H", "2B", "3B", "HR", "RBI", "BB", "SO", "SB", "CS"]]

    out.to_sql("padres_batting_logs", conn, if_exists="replace", index=False)
    print(f"  loaded {len(out):,} batting game logs")


def build_pitching(conn, names):
    cols = ["id", "team", "stattype", "p_ipouts", "p_h", "p_er", "p_k",
            "p_w", "p_bfp", "p_hbp", "wp", "lp", "save", "date", "opp", "vishome", "gametype"]
    rows = []
    for chunk in pd.read_csv("pitching.csv", usecols=cols, chunksize=CHUNKSIZE, low_memory=False):
        sub = chunk[(chunk["team"].isin(PADRES_TEAMS)) & (chunk["stattype"] == "value")]
        if not sub.empty:
            rows.append(sub)
        print(f"  pitching: scanned {chunk.shape[0]:,} rows, kept {sub.shape[0]}", end="\r")

    print()
    padres_p = pd.concat(rows, ignore_index=True)
    padres_p["player_name"] = padres_p["id"].map(names).fillna(padres_p["id"])
    padres_p["Date"] = padres_p["date"].apply(fmt_date)
    padres_p["Season"] = padres_p["Date"].str[:4].astype(int)
    padres_p["Outs"] = padres_p["p_ipouts"]
    padres_p["IP"] = (padres_p["p_ipouts"] / 3).round(1)
    # wp/lp/save are per-row boolean flags (1.0 if *this* pitcher got the decision),
    # not opponent-comparable ids -- NOT the same shape as batting/pitching "id" columns.
    padres_p["W"] = padres_p["wp"].fillna(0).astype(int)
    padres_p["L"] = padres_p["lp"].fillna(0).astype(int)
    padres_p["SV"] = padres_p["save"].fillna(0).astype(int)
    padres_p["home"] = padres_p["vishome"] == 1

    out = padres_p.rename(columns={
        "p_h": "H", "p_er": "ER", "p_k": "SO", "p_w": "BB", "p_bfp": "BFP", "p_hbp": "HBP",
        "opp": "Opponent", "gametype": "GameType",
    })[["player_name", "Date", "Season", "team", "Opponent", "home", "GameType",
        "IP", "Outs", "H", "ER", "SO", "BB", "BFP", "HBP", "W", "L", "SV"]]

    out.to_sql("padres_pitching_logs", conn, if_exists="replace", index=False)
    print(f"  loaded {len(out):,} pitching game logs")


def build_games(conn):
    """One row per Padres game: result + opponent + game type (regular/postseason)."""
    cols = ["gid", "team", "stattype", "date", "opp", "vishome", "win", "loss", "tie", "gametype"]
    rows = []
    for chunk in pd.read_csv("batting.csv", usecols=cols, chunksize=CHUNKSIZE, low_memory=False):
        sub = chunk[(chunk["team"].isin(PADRES_TEAMS)) & (chunk["stattype"] == "value")]
        if not sub.empty:
            rows.append(sub.drop_duplicates("gid"))
        print(f"  games: scanned {chunk.shape[0]:,} rows", end="\r")

    print()
    games = pd.concat(rows, ignore_index=True).drop_duplicates("gid")
    games["Date"] = games["date"].apply(fmt_date)
    games["Season"] = games["Date"].str[:4].astype(int)
    games["home"] = games["vishome"] == 1

    out = games.rename(columns={
        "opp": "Opponent", "win": "Win", "loss": "Loss", "tie": "Tie", "gametype": "GameType",
    })[["Date", "Season", "Opponent", "home", "GameType", "Win", "Loss", "Tie"]].sort_values("Date")

    out.to_sql("padres_games", conn, if_exists="replace", index=False)
    print(f"  loaded {len(out):,} games")


def build_players(conn, names):
    """One row per player: bio info, Padres tenure, primary position."""
    cols = ["id", "team", "bat", "throw", "season", "first_g", "last_g", "g"] + list(POSITION_COLS.keys())
    players = pd.read_csv("allplayers.csv", usecols=cols, dtype={"id": str, "team": str, "bat": str, "throw": str})
    sdn = players[players["team"].isin(PADRES_TEAMS)].copy()

    agg = sdn.groupby("id").agg(
        first_season=("season", "min"),
        last_season=("season", "max"),
        seasons_with_padres=("season", "nunique"),
        total_games=("g", "sum"),
        bat=("bat", "first"),
        throw=("throw", "first"),
        **{c: (c, "sum") for c in POSITION_COLS},
    ).reset_index()

    def primary_position(row):
        counts = {label: row[col] for col, label in POSITION_COLS.items()}
        return max(counts, key=counts.get)

    agg["primary_position"] = agg.apply(primary_position, axis=1)
    agg["player_name"] = agg["id"].map(names).fillna(agg["id"])

    out = agg[["player_name", "bat", "throw", "primary_position",
               "first_season", "last_season", "seasons_with_padres", "total_games"]] \
        .rename(columns={"bat": "Bats", "throw": "Throws"})

    out.to_sql("padres_players", conn, if_exists="replace", index=False)
    print(f"  loaded {len(out):,} players")


AGGREGATE_TABLE_NAMES = [
    "padres_season_standings", "padres_postseason_games",
    "padres_season_batting", "padres_career_batting",
    "padres_season_pitching", "padres_career_pitching",
]


def create_aggregate_tables(conn):
    # a prior run may have created these as VIEWs; DROP TABLE can't remove a view
    for name in AGGREGATE_TABLE_NAMES:
        row = conn.execute(
            "SELECT type FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()
        if row:
            conn.execute(f"DROP {row[0].upper()} IF EXISTS {name};")

    conn.executescript("""
    CREATE TABLE padres_season_standings AS
    SELECT Season,
           SUM(Win) AS Wins, SUM(Loss) AS Losses, SUM(Tie) AS Ties,
           COUNT(*) AS Games,
           ROUND(SUM(Win) * 1.0 / NULLIF(SUM(Win) + SUM(Loss), 0), 3) AS WinPct
    FROM padres_games
    WHERE GameType = 'regular'
    GROUP BY Season;

    DROP TABLE IF EXISTS padres_postseason_games;
    CREATE TABLE padres_postseason_games AS
    SELECT * FROM padres_games WHERE GameType != 'regular';

    DROP TABLE IF EXISTS padres_season_batting;
    CREATE TABLE padres_season_batting AS
    SELECT player_name, Season,
           COUNT(*) AS Games, SUM(AB) AS AB, SUM(H) AS H, SUM("2B") AS Doubles,
           SUM("3B") AS Triples, SUM(HR) AS HR, SUM(RBI) AS RBI, SUM(BB) AS BB,
           SUM(SO) AS SO, SUM(SB) AS SB, SUM(CS) AS CS,
           ROUND(SUM(H) * 1.0 / NULLIF(SUM(AB), 0), 3) AS AVG
    FROM padres_batting_logs
    GROUP BY player_name, Season;

    DROP TABLE IF EXISTS padres_career_batting;
    CREATE TABLE padres_career_batting AS
    SELECT player_name,
           COUNT(*) AS Games, SUM(AB) AS AB, SUM(H) AS H, SUM("2B") AS Doubles,
           SUM("3B") AS Triples, SUM(HR) AS HR, SUM(RBI) AS RBI, SUM(BB) AS BB,
           SUM(SO) AS SO, SUM(SB) AS SB, SUM(CS) AS CS,
           ROUND(SUM(H) * 1.0 / NULLIF(SUM(AB), 0), 3) AS AVG
    FROM padres_batting_logs
    GROUP BY player_name;

    DROP TABLE IF EXISTS padres_season_pitching;
    CREATE TABLE padres_season_pitching AS
    SELECT player_name, Season,
           COUNT(*) AS Games, ROUND(SUM(Outs) / 3.0, 1) AS IP, SUM(H) AS H,
           SUM(ER) AS ER, SUM(SO) AS SO, SUM(BB) AS BB,
           SUM(W) AS W, SUM(L) AS L, SUM(SV) AS SV,
           ROUND(9.0 * SUM(ER) / NULLIF(SUM(Outs) / 3.0, 0), 2) AS ERA
    FROM padres_pitching_logs
    GROUP BY player_name, Season;

    DROP TABLE IF EXISTS padres_career_pitching;
    CREATE TABLE padres_career_pitching AS
    SELECT player_name,
           COUNT(*) AS Games, ROUND(SUM(Outs) / 3.0, 1) AS IP, SUM(H) AS H,
           SUM(ER) AS ER, SUM(SO) AS SO, SUM(BB) AS BB,
           SUM(W) AS W, SUM(L) AS L, SUM(SV) AS SV,
           ROUND(9.0 * SUM(ER) / NULLIF(SUM(Outs) / 3.0, 0), 2) AS ERA
    FROM padres_pitching_logs
    GROUP BY player_name;
    """)
    print("  created 6 aggregate tables")


def main():
    print("Loading player names...")
    names = load_player_names()

    conn = sqlite3.connect("padres_history.db")
    for tbl in ["padres_batting_logs", "padres_pitching_logs", "padres_games", "padres_players"]:
        conn.execute(f"DROP TABLE IF EXISTS {tbl};")

    print("Building batting logs...")
    build_batting(conn, names)

    print("Building pitching logs...")
    build_pitching(conn, names)

    print("Building games (results/standings/postseason)...")
    build_games(conn)

    print("Building player bios...")
    build_players(conn, names)

    print("Creating aggregate tables...")
    create_aggregate_tables(conn)

    conn.execute("CREATE INDEX idx_batting_name ON padres_batting_logs(player_name);")
    conn.execute("CREATE INDEX idx_batting_date ON padres_batting_logs(Date);")
    conn.execute("CREATE INDEX idx_batting_season ON padres_batting_logs(Season);")
    conn.execute("CREATE INDEX idx_pitching_name ON padres_pitching_logs(player_name);")
    conn.execute("CREATE INDEX idx_pitching_date ON padres_pitching_logs(Date);")
    conn.execute("CREATE INDEX idx_pitching_season ON padres_pitching_logs(Season);")
    conn.execute("CREATE INDEX idx_games_date ON padres_games(Date);")
    conn.execute("CREATE INDEX idx_games_season ON padres_games(Season);")
    conn.execute("CREATE INDEX idx_players_name ON padres_players(player_name);")
    conn.execute("CREATE INDEX idx_season_batting ON padres_season_batting(player_name, Season);")
    conn.execute("CREATE INDEX idx_season_pitching ON padres_season_pitching(player_name, Season);")
    conn.execute("CREATE INDEX idx_career_batting ON padres_career_batting(player_name);")
    conn.execute("CREATE INDEX idx_career_pitching ON padres_career_pitching(player_name);")
    conn.commit()

    cur = conn.cursor()
    all_tables = ["padres_batting_logs", "padres_pitching_logs", "padres_games", "padres_players",
                  "padres_season_standings", "padres_postseason_games",
                  "padres_season_batting", "padres_season_pitching",
                  "padres_career_batting", "padres_career_pitching"]
    for tbl in all_tables:
        cur.execute(f"SELECT COUNT(*) FROM {tbl};")
        print(f"{tbl}: {cur.fetchone()[0]:,} rows")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
