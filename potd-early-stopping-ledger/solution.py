from typing import List, Tuple


def early_stopping_ledger(losses: List[float], patience: int) -> Tuple[int, float]:
    """
    losses: validation losses in epoch order, length n.
    patience: number of consecutive non-improving epochs that triggers a stop.

    Track the best loss seen so far (starts at +infinity) and a
    non-improving counter. An epoch improves iff its loss is strictly less
    than the best so far: reset the counter and update best on improvement,
    otherwise increment the counter. Stop at the first epoch where the
    counter reaches patience. If that never happens, stop after the final
    epoch. Return (1-based stop epoch, best loss observed through that
    epoch).
    """
    loss = losses[0]
    pat_run = patience

    for i in range(1, len(losses)):
        if losses[i] < loss:
            loss = losses[i]
            pat_run = patience
        else:
            pat_run -= 1
            if pat_run == 0:
                return (i + 1, loss)

    return (len(losses), loss)
