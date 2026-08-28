"""Reproduce the roster comparison: quality table, then optimized vs greedy.

    python main.py                 # the headline comparison
    python main.py --sweep         # sensitivity to the synergy prior
    python main.py --verify        # check the MILP against brute force
    python main.py --accuracy      # cross-validated accuracy of the win models
"""

import argparse

import pandas as pd

from roster.data import ROLES, RowsByRole, load_players, split_by_role
from roster.optimize import Roster, brute_force_roster, greedy_roster, optimal_roster
from roster.quality import QualityByRole, role_accuracy, role_quality
from roster.synergy import DEFAULT_SHRINKAGE, pair_synergy, synergy_lookup

ROLE_LABEL: dict[str, str] = {
    "top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support",
}


def build(shrinkage: float) -> tuple[RowsByRole, QualityByRole, pd.DataFrame]:
    by_role = split_by_role(load_players())
    quality = {role: role_quality(by_role[role]) for role in ROLES}
    pairs = pair_synergy(by_role, shrinkage=shrinkage)
    return by_role, quality, pairs


def print_quality_table(quality: QualityByRole, n: int = 3) -> None:
    print(f"\nTop {n} players by quality score, per role")
    for role in ROLES:
        best = quality[role].sort_values(ascending=False).head(n)
        listed = ", ".join(f"{name} {value:+.3f}" for name, value in best.items())
        print(f"  {ROLE_LABEL[role]:<8} {listed}")


def print_comparison(optimized: Roster, greedy: Roster) -> None:
    print("\n                 Optimized            Greedy")
    for role in ROLES:
        left = f"{optimized.players[role]}"
        right = f"{greedy.players[role]}"
        print(f"  {ROLE_LABEL[role]:<8} {left:<20} {right}")
    print(f"  {'quality':<8} {optimized.quality:<20.3f} {greedy.quality:.3f}")
    print(f"  {'synergy':<8} {optimized.synergy:<20.3f} {greedy.synergy:.3f}")
    print(f"  {'total':<8} {optimized.total:<20.3f} {greedy.total:.3f}")
    print(
        f"\n  Optimizing gives up {greedy.quality - optimized.quality:.3f} quality "
        f"to gain {optimized.synergy - greedy.synergy:.3f} synergy, "
        f"a net {optimized.total - greedy.total:+.3f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE,
                        help="prior games pulling pair win rates toward even (default: %(default)s)")
    parser.add_argument("--sweep", action="store_true", help="rerun across a range of priors")
    parser.add_argument("--verify", action="store_true", help="also solve by brute force and compare")
    parser.add_argument("--accuracy", action="store_true", help="report win-model accuracy per role")
    parser.add_argument("--log", action="store_true", help="show the solver log")
    args = parser.parse_args()

    by_role, quality, pairs = build(args.shrinkage)

    print(f"{sum(len(q) for q in quality.values())} players "
          f"({', '.join(f'{len(quality[r])} {r}' for r in ROLES)}), "
          f"{len(pairs)} pairs with shared games, prior k={args.shrinkage:g}")

    if args.accuracy:
        print("\nWin model, 5-fold accuracy grouped by game")
        for role in ROLES:
            print(f"  {ROLE_LABEL[role]:<8} {role_accuracy(by_role[role]):.3f}")

    print_quality_table(quality)

    synergy = synergy_lookup(pairs)
    optimized, model = optimal_roster(quality, synergy, log=args.log)
    print_comparison(optimized, greedy_roster(quality, synergy))

    if args.verify:
        checked = brute_force_roster(quality, synergy)
        agree = abs(checked.total - optimized.total) < 1e-9
        print(f"\n  Brute force over every roster: {checked.total:.6f} "
              f"vs solver {optimized.total:.6f} -> {'agree' if agree else 'DISAGREE'}")
        print(f"  Model size: {model.NumVars} variables, {model.NumConstrs} constraints")

    if args.sweep:
        print("\nSensitivity to the synergy prior")
        print("  k     roster                                     quality  synergy    total     gain")
        for k in (0, 5, 10, 20, 50, 100):
            _, q_k, pairs_k = build(k) if k != args.shrinkage else (by_role, quality, pairs)
            syn_k = synergy_lookup(pairs_k)
            best, _ = optimal_roster(q_k, syn_k)
            base = greedy_roster(q_k, syn_k)
            names = "/".join(best.players[r] for r in ROLES)
            print(f"  {k:<5g} {names:<42} {best.quality:7.3f}  {best.synergy:7.3f}  "
                  f"{best.total:7.3f}  {best.total - base.total:+.3f}")


if __name__ == "__main__":
    main()
