"""Pairwise synergy, estimated from how often two players win together."""

from itertools import combinations

import pandas as pd

from .data import ROLES, RowsByRole

# (player_a, role_a, player_b, role_b), roles always in ROLES order so a pair
# has exactly one key.
PairKey = tuple[str, str, str, str]
SynergyMap = dict[PairKey, float]

DEFAULT_SHRINKAGE: float = 20.0


def _shared_games(by_role: RowsByRole, role_a: str, role_b: str) -> pd.DataFrame:
    """Team-games where both of these roles were filled by modeled players.

    Joining on (gameid, result) keeps teammates together, since the two teams in
    a game have opposite results. Only these two roles are joined: joining all
    five would drop a game whenever any other player fell outside the pool,
    losing most of the evidence for reasons unrelated to the pair.
    """
    left = by_role[role_a][["gameid", "result", "playername"]].rename(columns={"playername": role_a})
    right = by_role[role_b][["gameid", "result", "playername"]].rename(columns={"playername": role_b})
    return left.merge(right, on=["gameid", "result"])


def pair_synergy(by_role: RowsByRole, shrinkage: float = DEFAULT_SHRINKAGE) -> pd.DataFrame:
    """Synergy for every pair of players who have shared a roster.

    One row per observed pair, with the win rate above even shrunk toward zero
    by a prior of `shrinkage` imaginary even games:

        s = (wins + k/2) / (games + k) - 1/2

    Without it a single shared game scores the maximum +/- 0.5 off a coin flip,
    while a well-observed pair barely moves. Pairs that never played together
    are absent, and the optimizer reads them as 0 -- ignorance, not neutrality.
    """
    rows: list[dict] = []
    for role_a, role_b in combinations(ROLES, 2):
        shared = _shared_games(by_role, role_a, role_b)
        grouped = shared.groupby([role_a, role_b])["result"].agg(["sum", "size"])
        for (player_a, player_b), record in grouped.iterrows():
            wins, games = record["sum"], record["size"]
            rows.append({
                "player_a": player_a, "role_a": role_a,
                "player_b": player_b, "role_b": role_b,
                "games": int(games),
                "win_rate": wins / games,
                "synergy": (wins + shrinkage / 2) / (games + shrinkage) - 0.5,
            })
    return pd.DataFrame(rows)


def synergy_lookup(pairs: pd.DataFrame) -> SynergyMap:
    """{(player_a, role_a, player_b, role_b): synergy}, for the optimizer."""
    return {
        (r.player_a, r.role_a, r.player_b, r.role_b): r.synergy
        for r in pairs.itertuples()
    }
