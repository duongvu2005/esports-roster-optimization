"""Load the Oracle's Elixir dumps and reduce them to the player-seasons we model."""

from pathlib import Path

import pandas as pd

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
SEASONS: tuple[int, ...] = (2023, 2024, 2025)
LEAGUE: str = "LCK"
ROLES: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")

# Player rows keyed by role.
RowsByRole = dict[str, pd.DataFrame]

# Fed to the quality model. Excludes anything identifying the match or opponent.
STATS: list[str] = [
    "kills", "deaths", "assists",
    "teamkills", "teamdeaths",
    "doublekills", "triplekills", "quadrakills", "pentakills",
    "firstbloodvictim",
    "team kpm", "ckpm",
    "damagetochampions", "dpm", "damageshare",
    "damagetakenperminute", "damagemitigatedperminute", "damagetotowers",
    "wardsplaced", "wpm", "wardskilled", "wcpm", "controlwardsbought",
    "visionscore", "vspm",
    "totalgold", "earnedgold", "earned gpm", "earnedgoldshare", "goldspent",
    "total cs", "minionkills", "monsterkills", "cspm",
]


def _season_path(year: int) -> Path:
    return DATA_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def _load_season(year: int) -> pd.DataFrame:
    """One season, restricted to complete LCK player rows."""
    path = _season_path(year)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download the {year} match data from "
            "https://oracleselixir.com/tools/downloads into data/ "
            "(see README, Getting the data)."
        )
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["league"] == LEAGUE) & (df["datacompleteness"] == "complete")]
    # Drops the two team-summary rows each game carries beside its ten players.
    df = df[df["position"].isin(ROLES)]
    return df.dropna(subset=["playername"]).copy()


def load_players() -> pd.DataFrame:
    """Player-match rows for every player who appeared in all three seasons.

    Everything downstream averages over a player's own history, so a player with
    a handful of games would score on sampling noise. Three seasons is a blunt
    proxy for having enough history to estimate.
    """
    seasons = [_load_season(y) for y in SEASONS]
    veterans = set.intersection(*(set(s["playername"].unique()) for s in seasons))
    seasons = [s[s["playername"].isin(veterans)].copy() for s in seasons]

    # Two positions means two statistical profiles, which the per-role models
    # cannot represent. Only a couple of players, so drop rather than model.
    roles_per_player = pd.concat(seasons).groupby("playername")["position"].nunique()
    switchers = roles_per_player[roles_per_player > 1].index
    seasons = [s[~s["playername"].isin(switchers)] for s in seasons]

    return pd.concat(seasons, ignore_index=True)


def split_by_role(df: pd.DataFrame) -> RowsByRole:
    """{role: rows}. Rows are left whole; the quality model drops what it needs to."""
    return {r: df[df["position"] == r].copy() for r in ROLES}
