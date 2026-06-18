"""Load and prepare data for training."""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .preprocess import preprocess


class RatingsDataset(Dataset):
    """PyTorch Dataset for user-book ratings."""

    def __init__(self, users, books, ratings):
        self.users = users
        self.books = books
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.books[idx], self.ratings[idx]


def load_data(path: str = None, test_size: float = 0.2, batch_size: int = 64):
    """Run full data loading pipeline.

    :param path: path to ratings csv
    :param test_size: proportion of data for testing
    :param batch_size: batch size for DataLoader
    :return: train_loader, test_loader, num_users, num_books
    """

    # retrieve sparse matrix from preprocess pipeline
    sparse = preprocess(path) if path else preprocess()

    # convert CSR to COO to get (user, book, rating) triplets
    coo = sparse.tocoo()

    users = torch.tensor(coo.row, dtype=torch.long)
    books = torch.tensor(coo.col, dtype=torch.long)
    ratings = torch.tensor(coo.data, dtype=torch.float32)

    # get num_users and num_books for embedding layers
    num_users = users.max().item() + 1
    num_books = books.max().item() + 1

    # split into train/test
    train_idx, test_idx = train_test_split(
        range(len(ratings)), test_size=test_size, random_state=42
    )

    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)

    # create datasets
    train_dataset = RatingsDataset(
        users[train_idx], books[train_idx], ratings[train_idx]
    )
    test_dataset = RatingsDataset(users[test_idx], books[test_idx], ratings[test_idx])

    # wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, num_users, num_books


if __name__ == "__main__":
    train_loader, test_loader, num_users, num_books = load_data()
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Num users: {num_users} | Num Books: {num_books}")
