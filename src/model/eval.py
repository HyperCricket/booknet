"""Eval.py - evaluate the trained recommender checkpoint on GoodBooks-10k.


Computes 4 things:
    RMSE (root mean squared error) - rating-prediction on the held-out test split.
    (When the model guesses a star rating, how far off is it?) Lower loss is better.

    HR@10 (hit rate) - This asks about rankings. Takes a book the user liked (in the test set) and has the model score and rank every book in the catalog. Did it appear in the top 10?

    NDCG@10 (normalized discounted cumulative gain @ position 10) - same as HR, but cares about where in the top 10 the book landed.

    Coverage -  "Across all users, how much of the catalog does the model ever other recommending? Prevents the recommender from just recommending the top 50 most popular books

"""

import math
from pathlib import Path
from collections import defaultdict
import torch
from model.mf_baseline import MatrixFactorization
from data.dataset import load_data

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_model(num_users, num_items):
    return MatrixFactorization(num_users, num_items, embedding_dim=50)


CHECKPOINT = REPO_ROOT / "models" / "mf_baseline.pt"
RESULTS_FILE = REPO_ROOT / "results" / "mf_baseline.txt"
K = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def compute_rmse(model, test_loader):
    """Rating-prediction error over the held-out set. Iterates the test Dataloader directly, reusing its batching. Metric for MSE loss."""
    se, n = 0.0, 0
    for users, books, ratings in test_loader:
        u = users.to(DEVICE)
        b = books.to(DEVICE)
        r = ratings.to(DEVICE)
        preds = model(u, b).squeeze(-1)
        se += torch.sum((preds - r) ** 2).item()
        n += len(r)
    return math.sqrt(se / n)


def collect_items_by_user(loader):
    """Flatten a Dataloader into {user_id: [book_id, ...]}. Used to know which books each user saw in training (to mask) and which were held out (to find)."""
    by_user = defaultdict(list)
    for users, books, _ratings in loader:
        for u, b in zip(users.tolist(), books.tolist()):
            by_user[u].append(b)
    return by_user


@torch.no_grad()
def score_all_items(model, user_id, num_items, batch=20000):
    """Score every book in the catalog for one user. Returns a [num_items] tensor. Batched in case catalog is large."""
    all_items = torch.arange(num_items, device=DEVICE)
    out = torch.empty(num_items, device=DEVICE)
    for i in range(0, num_items, batch):
        chunk = all_items[i : i + batch]
        u = torch.full_like(chunk, user_id)
        out[i : i + batch] = model(u, chunk).squeeze(-1)
    return out


@torch.no_grad()
def compute_ranking(model, train_by_user, test_by_user, num_items, k=K):
    """Full catalog ranking: for each held-out (user, book) score the whole catalog, mask books the user already saw in training, and check where the held-out book lands. Averages HR@k / NDCG@k and tracks coverage.

    Each user is scored once - cheaper when user has several test interactions.

    """
    hits, ndcgs = [], []
    recommended = set()

    for user, positives in test_by_user.items():
        scores = score_all_items(model, user, num_items)

        seen = train_by_user.get(user, [])

        if seen:
            scores[seen] = float("-inf")  # don't recommend already seen books

        # Coverage: what this user would actually be shown
        topk = torch.topk(scores, k).indices.tolist()
        recommended.update(topk)

        for pos in positives:
            rank = int((scores > scores[pos]).sum().item()) + 1
            if rank <= k:
                hits.append(1.0)
                ndcgs.append(1.0 / math.log2(rank + 1))
            else:
                hits.append(0.0)
                ndcgs.append(0.0)

    hr = sum(hits) / len(hits)
    ndcg = sum(ndcgs) / len(ndcgs)
    coverage = len(recommended) / num_items
    return hr, ndcg, coverage


def main():
    print(f"Device: {DEVICE}")

    train_loader, test_loader, num_users, num_items = load_data()
    print(f"Users: {num_users} Items: {num_items}")
    model = build_model(num_users, num_items).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")
    rmse = compute_rmse(model, test_loader)
    print(f"train RMSE (sanity check): {compute_rmse(model, train_loader):.4f}")

    train_by_user = collect_items_by_user(train_loader)
    test_by_user = collect_items_by_user(test_loader)
    hr, ndcg, coverage = compute_ranking(model, train_by_user, test_by_user, num_items)

    report = (
        f"RMSE      : {rmse:.4f}\n"
        f"HR@{K}    : {hr:.4f}\n"
        f"NDCG@{K}  : {ndcg:.4f}\n"
        f"Coverage  : {coverage:.4f}\n"
    )

    print("\n" + report)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(report)
    print(f"Wrote results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
