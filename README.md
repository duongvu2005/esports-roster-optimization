# Synergy or Star Power

Do you build a League of Legends roster by signing the best player at every position, or by
signing players who win together? This picks the roster both ways from three seasons of LCK
match data and measures what the trade is worth.

The short answer, on 2023–2025 LCK: chemistry is worth paying for. The optimizer keeps the
Kiin–Canyon–Chovy core that the greedy pick also takes, then rebuilds the bot side around
players who have history with it — giving up **0.709 points of individual quality to buy
0.916 points of synergy**, a net gain of 0.207.

```
                 Optimized            Greedy
  Top      Kiin                 Kiin
  Jungle   Canyon               Canyon
  Mid      Chovy                Chovy
  Bot      Aiming               Gumayusi
  Support  Lehends              Delight
  quality  5.685                6.393
  synergy  1.908                0.992
  total    7.593                7.386
```

Eight of the optimized roster's ten pairs have actually played together; four of greedy's do.

Term project for **6.C571 Optimization**, MIT, Fall 2025. The five-page write-up is in
[`docs/report.pdf`](docs/report.pdf); its results correspond to `python main.py --shrinkage 0`.

## The model

Roster construction is a subset selection problem: pick exactly one player for each of the
five roles, maximizing individual quality plus the synergy of every pair you end up with.

```math
\max \sum_{r}\sum_{i} q_{ir}x_{ir} \;+\; \sum_{r \lt r'}\sum_{i,j} s_{irjr'}x_{ir}x_{jr'}
\qquad \mathrm{s.t.} \quad\sum_i x_{ir} = 1 \;\; \forall r
```

The objective is quadratic, because a synergy term is earned only when *both* of its players
are selected. It is linearized with one binary $`y_{irjr'}`$ per pair and the three constraints
in `roster/optimize.py` — the two upper bounds stop the solver claiming a positive synergy it
has not bought both players for, and the lower bound stops it disowning a negative one it has.

**Quality** comes from a per-role logistic regression of win/loss on 34 per-match statistics.
A player's score is their average standardized stat line pushed through the fitted weights,
$`q = \beta^\top\bar{x}`$, then standardized within the role so a quality point means one
standard deviation among that role's players. One model per role, because the stat line that
wins games looks nothing alike for a support and a bot laner.

**Synergy** is a pair's win rate above even, shrunk toward zero by a prior of `k` imaginary
even games:

```math
s = \frac{\text{wins} + k/2}{\text{games} + k} - \frac{1}{2}
```

Ninety-nine of the 430 possible pairs have ever shared a roster, and ten of those have played
exactly one game together — where an unshrunk win rate is a coin flip reported to three
decimals, and indeed those ten average the maximum possible $`|s| = 0.5`$ against 0.132 for
pairs with more than twenty games. The prior costs a well-observed pair almost nothing (a
224-game pair keeps 92% of its estimate at `k = 20`) while collapsing the one-game pairs to
near zero.

A pair is scored by joining only the two roles it involves. Joining all five at once would drop
a game whenever *any* of the other three players fell outside the modeled pool, losing most of
the evidence for reasons unrelated to the pair being scored.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

The model is small enough (132 variables) to solve under the size-limited license bundled
with `pip install gurobipy`; no academic license needed.

```bash
python main.py --verify      # check the solver against brute force over all 10,752 rosters
python main.py --sweep       # sensitivity to the synergy prior
python main.py --accuracy    # cross-validated accuracy of the win models
python main.py --shrinkage 0 # raw win rates, no shrinkage
```

### Getting the data

Three CSVs from [Oracle's Elixir](https://oracleselixir.com/tools/downloads), dropped into
`data/`:

```
data/2023_LoL_esports_match_data_from_OraclesElixir.csv
data/2024_LoL_esports_match_data_from_OraclesElixir.csv
data/2025_LoL_esports_match_data_from_OraclesElixir.csv
```

They are ~240 MB together and are not committed here. From those, the pipeline keeps LCK
games flagged `complete`, then keeps only the 33 players who appeared in all three seasons —
both estimates are averages over a player's own history, so a player with a handful of games
gets a score driven by sampling noise rather than by skill.

## How robust is it?

The chosen roster does not move until the synergy prior is heavy enough to erase synergy
altogether, which is the reassuring direction:

| prior `k` | roster | quality | synergy | gain over greedy |
|---|---|---|---|---|
| 0 | Kiin/Canyon/Chovy/Aiming/Lehends | 5.685 | 2.167 | +0.358 |
| 10 | Kiin/Canyon/Chovy/Aiming/Lehends | 5.685 | 2.029 | +0.277 |
| 20 | Kiin/Canyon/Chovy/Aiming/Lehends | 5.685 | 1.908 | +0.207 |
| 50 | Kiin/Canyon/Chovy/Aiming/Lehends | 5.685 | 1.626 | +0.050 |
| 100 | Kiin/Canyon/Chovy/Viper/Delight | 6.291 | 0.832 | +0.010 |

`--verify` solves the same problem by enumerating all 8×4×8×6×7 rosters and checks the two
answers agree, so the MILP is validated against an implementation with no modeling in it.

## Known limitations

These are real and I would rather state them than have them found.

**The win model is far too good.** Cross-validated accuracy, grouped so a game cannot be
split across folds, is 0.978–0.986. Nothing predicts a League game that well from a player's
own contribution — the feature list includes team-level and end-state statistics
(`teamkills`, `teamdeaths`, `team kpm`, `ckpm`, `totalgold`, `damagetotowers`), so the model
is substantially reading the scoreboard rather than the player. Dropping those features takes
accuracy to about 0.78, which is a more believable number for a genuine skill signal. Quality
here should be read as "plays in winning games", with the causal arrow left undetermined.

**Synergy is confounded with team quality.** A pair's win rate together absorbs everything
about the organization they played for — coaching, drafting, the other three players. It is
not a clean interaction effect, and strong players are systematically paired with other
strong players. The optimizer's preference for the Kiin–Canyon–Chovy core is partly a
statement that Gen.G won a lot of games.

**The benchmark is in-sample.** The optimizer maximizes `quality + synergy` and is then
declared the winner on `quality + synergy`; it cannot lose. The number worth reading is not
"optimization wins" but the exchange rate it reveals — how much individual talent this data
says chemistry is worth. A genuine test would fit on 2023–24 and ask whether the score
ordering predicts real 2025 results.

**Unobserved pairs are scored zero, which is ignorance, not neutrality.** Two of the optimized
roster's pairs and six of greedy's have never shared a game, and a model that treated missing
history as a mild negative rather than as nothing would choose differently.

**The candidate pool is tiny.** Requiring three full seasons leaves 33 players, and only four
junglers. That is the price of wanting estimates that mean something, but it means the search
space is 10,752 rosters — small enough that the MILP is a demonstration of the formulation
rather than a computational necessity.

## Credits

Match data from [Oracle's Elixir](https://oracleselixir.com/), maintained by Tim Sevenhuysen,
used with attribution as its terms ask. The data is not redistributed in this repository, and
the license below covers this code only, not the data.

## License

MIT. See [LICENSE](LICENSE).
