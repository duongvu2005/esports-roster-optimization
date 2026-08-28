"""Individual player quality, estimated from how a player's stat line predicts a win."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from .data import STATS

# Per role, a Series of scores indexed by player name.
QualityByRole = dict[str, pd.Series]


def _fit_role(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, LogisticRegression]:
    """Drop incomplete stat lines, standardize, fit win/loss.

    Returns the surviving rows too, so callers line up with what the model saw.
    """
    rows = rows.dropna(subset=STATS + ["result"])
    standardized = StandardScaler().fit_transform(rows[STATS])
    model = LogisticRegression(max_iter=2000).fit(standardized, rows["result"].values)
    return rows, standardized, model


def role_quality(rows: pd.DataFrame) -> pd.Series:
    """Quality scores for one role, as a Series indexed by player name.

    One model per role, since the stat line that wins games differs completely
    between a support and a bot laner. A player scores their average
    standardized stat line through the fitted weights, q = b . xbar. The
    intercept is dropped; it is constant within a role and cancels below.
    """
    rows, standardized, model = _fit_role(rows)

    frame = pd.DataFrame(standardized, columns=STATS)
    frame["playername"] = rows["playername"].values
    per_player = frame.groupby("playername")[STATS].mean()

    raw = pd.Series(per_player.values.dot(model.coef_.ravel()), index=per_player.index)
    # Comparable only within a role, so a quality point means one standard
    # deviation among this role's players -- which also makes it addable to synergy.
    return (raw - raw.mean()) / raw.std()


def role_accuracy(rows: pd.DataFrame) -> float:
    """Accuracy of the win model, cross-validated with games kept whole.

    Uncomfortably high; see "Known limitations" in the README.
    """
    rows, standardized, _ = _fit_role(rows)
    scores = cross_val_score(
        LogisticRegression(max_iter=2000),
        standardized,
        rows["result"].values,
        cv=GroupKFold(5),
        groups=rows["gameid"],
        scoring="accuracy",
    )
    return float(scores.mean())
