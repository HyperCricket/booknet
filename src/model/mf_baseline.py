import torch
import torch.nn as nn
import torch.nn.init as init


class MatrixFactorization(nn.Module):
    """Matrix Factorization Model."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()

        self.user_embeddings = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim
        )
        self.item_embeddings = nn.Embedding(
            num_embeddings=num_items, embedding_dim=embedding_dim
        )

        init.normal_(self.user_embeddings.weight, std=0.1)
        init.normal_(self.item_embeddings.weight, std=0.1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Perform one dot product at a time, element wise."""

        user_embeddings = self.user_embeddings(users)
        item_embeddings = self.item_embeddings(items)

        out = (user_embeddings * item_embeddings).sum(dim=1)

        return out
