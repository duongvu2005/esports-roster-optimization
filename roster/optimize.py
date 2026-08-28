"""Selecting the roster: the MILP, the greedy benchmark, and a brute-force check."""

from dataclasses import dataclass
from itertools import combinations, product

import gurobipy as gp
from gurobipy import GRB

from .data import ROLES
from .quality import QualityByRole
from .synergy import SynergyMap


@dataclass(frozen=True)
class Roster:
    players: dict[str, str]   # role -> player name
    quality: float            # summed individual quality
    synergy: float            # summed pairwise synergy

    @property
    def total(self) -> float:
        return self.quality + self.synergy


def score(players: dict[str, str], quality: QualityByRole, synergy: SynergyMap) -> Roster:
    """Score a chosen five: individual quality plus every pair that is scored."""
    q = sum(quality[role][players[role]] for role in ROLES)
    s = 0.0
    for role_a, role_b in combinations(ROLES, 2):
        s += synergy.get((players[role_a], role_a, players[role_b], role_b), 0.0)
    return Roster(players=dict(players), quality=q, synergy=s)


def greedy_roster(quality: QualityByRole, synergy: SynergyMap) -> Roster:
    """The benchmark: best available player in each role, ignoring interactions."""
    return score({role: quality[role].idxmax() for role in ROLES}, quality, synergy)


def optimal_roster(
    quality: QualityByRole, synergy: SynergyMap, log: bool = False
) -> tuple[Roster, gp.Model]:
    """Maximize quality + synergy over rosters, one player per role.

    A synergy term is earned only when both its players are picked, making the
    objective quadratic; it is linearized with a y_ij per pair and three
    constraints. All three matter: the upper bounds stop the solver claiming a
    positive synergy it has not bought, the lower bound stops it disowning a
    negative one. Only observed pairs get a variable, since the rest score 0.
    """
    model = gp.Model("roster")
    model.Params.OutputFlag = 1 if log else 0

    x = {
        (player, role): model.addVar(vtype=GRB.BINARY, name=f"x[{player},{role}]")
        for role in ROLES
        for player in quality[role].index
    }
    y = {
        key: model.addVar(vtype=GRB.BINARY, name="y[{},{},{},{}]".format(*key))
        for key in synergy
    }

    for role in ROLES:
        model.addConstr(
            gp.quicksum(x[player, role] for player in quality[role].index) == 1,
            name=f"one_{role}",
        )

    for (player_a, role_a, player_b, role_b) in synergy:
        pair = y[player_a, role_a, player_b, role_b]
        first, second = x[player_a, role_a], x[player_b, role_b]
        model.addConstr(pair <= first)
        model.addConstr(pair <= second)
        model.addConstr(pair >= first + second - 1)

    model.setObjective(
        gp.quicksum(quality[role][player] * x[player, role]
                    for role in ROLES for player in quality[role].index)
        + gp.quicksum(synergy[key] * y[key] for key in synergy),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"solver finished with status {model.Status}")

    chosen = {
        role: next(p for p in quality[role].index if x[p, role].X > 0.5)
        for role in ROLES
    }
    return score(chosen, quality, synergy), model


def brute_force_roster(quality: QualityByRole, synergy: SynergyMap) -> Roster:
    """Every legal roster, scored directly -- only ~11k, so this is cheap.

    Not part of the method; it checks the MILP against code with no modeling in it.
    """
    candidates = product(*(quality[role].index for role in ROLES))
    return max(
        (score(dict(zip(ROLES, combo)), quality, synergy) for combo in candidates),
        key=lambda r: r.total,
    )
