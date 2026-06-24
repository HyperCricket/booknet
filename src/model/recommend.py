"""recommend.py - generate book recommendations for a real user from the trained MF model.

For a given user it prints the books they rated highest next to the model's top-10 recommendations, so it's easy to tell whether the recommendations actually fit the users taste.

Run from inside src/:
    python -m model.recommend for random user
    python -m model.recommend 42 for specific user inside
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from model.mf_baseline import MatrixFactorization
from data.preprocess import load, clean, RATINGS_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "models" / "mf_baseline.pt"
BOOKS_PATH = RATINGS_PATH.parent / "books.csv"
EMBEDDING_DIM = 50
TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_mappings():
    """Recover the index <-> id mapping the model uses.

    preprocess() builds its matrix withpandas pivot table, which sorts the user_ids and book_ids ascending. So model index j corresponds to the j-th SMALLEST surviving id. We reproduce that exactordering with a sort. The mapping was never saved, so we recompute it from filtered data.

    """
    ratings = load(RATINGS_PATH)
    filtered = clean(ratings)

    book_ids = np.sort(filtered["book_id"].unique())
    user_ids = np.sort(filtered["user_id"].unique())

    book_id_to_index = {int(b): i for i, b in enumerate(book_ids)}
    user_id_to_index = {int(u): i for i, u in enumerate(user_ids)}

    return filtered, book_ids, user_ids, book_id_to_index, user_id_to_index


def load_titles():
    """Build {book_id: 'Title - Author'} from books.csv for display."""
    books = pd.read_csv(BOOKS_PATH)
    title_col = "title" if "title" in books.columns else "original_title"
    has_author = "authors" in books.columns

    titles = {}
    for _, row in books.iterrows():
        bid = int(row["book_id"])
        name = str(row[title_col])
        titles[bid] = f"{name} - {row['authors']}" if has_author else name
    return titles


@torch.no_grad()
def score_all_items(model, user_index, num_items):
    """Score every book in the catalog for one user. Returns a [num_items] tensor."""
    items = torch.arange(num_items, device=DEVICE)
    user_vec = model.user_embeddings(torch.tensor([user_index], device=DEVICE))
    item_vecs = model.item_embeddings(items)
    return (user_vec * item_vecs).sum(dim=1)


def main():
    filtered, book_ids, user_ids, book_id_to_index, user_id_to_index = build_mappings()
    num_users, num_items = len(user_ids), len(book_ids)
    print(f"Recovered mapping: {num_users}, {num_items} books")

    titles = load_titles()

    model = MatrixFactorization(num_users, num_items, embedding_dim=EMBEDDING_DIM).to(
        DEVICE
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # Pick a user (model index): from the command line, else choose randomly
    if len(sys.argv) > 1:
        u_index = int(sys.argv[1])
        if not (0 <= u_index < num_users):
            print(f"User index must be between 0 and {num_users - 1}.")
            return
    else:
        u_index = int(np.random.randint(num_users))

    original_user_id = int(user_ids[u_index])
    print(f"\nUser model-index {u_index} (original user_id {original_user_id})")

    # Everything the model has already rated - both their taste and the mask
    user_rows = filtered[filtered["user_id"] == original_user_id]
    rated_indices = [book_id_to_index[int(b)] for b in user_rows["book_id"]]

    # What they loved
    loved = user_rows.sort_values("rating", ascending=False).head(8)
    print("\n Books this user rated highest:")
    for _, r in loved.iterrows():
        bid = int(r["book_id"])
        print(f" {int(r['rating'])}* {titles.get(bid, f'book_id {bid}')}")

    # Score the whole catalog, hide already-read books, take top K
    scores = score_all_items(model, u_index, num_items)
    scores[rated_indices] = float("-inf")
    top_indices = torch.topk(scores, TOP_K).indices.tolist()

    print(f"\n Top {TOP_K} recommendations (not yet rated):")
    for rank, idx in enumerate(top_indices, 1):
        bid = int(book_ids[idx])
        print(
            f" {rank:2d}. {titles.get(bid, f'book_id {bid}')} (score {scores[idx]:.2f})"
        )


if __name__ == "__main__":
    main()
