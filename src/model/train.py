from model.mf_baseline import MatrixFactorization
from data.dataset import load_data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Setup:
train_loader, _, num_users, num_books = load_data()
model = MatrixFactorization(
    num_users, num_books, embedding_dim=50
)  # Embedding dim = hyperparameter
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 10

"""Training Loop

1. wipe out old gradients
2. forward pass
3. measure error
4. compute gradients
5. nudge numbers
"""

print("Starting Training Now!")

for epoch in range(num_epochs):
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} / {num_epochs}")

    for users, books, actual_ratings in tqdm(train_loader):
        optimizer.zero_grad()
        predicted_ratings = model(users, books)
        loss = loss_fn(predicted_ratings, actual_ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}: avg loss = {total_loss / len(train_loader):.4f}")
