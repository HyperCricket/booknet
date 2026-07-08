import torch
import torch.nn as nn
import torch.nn.init as init

class NCF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        init.normal_(self.user_embedding.weight, std = 0.1)
        init.normal_(self.item_embedding.weight, std = 0.1)


    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
