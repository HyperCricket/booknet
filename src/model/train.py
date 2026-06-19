import os
from model.mf_baseline import MatrixFactorization
from data.dataset import load_data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Setup:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, _, num_users, num_books = load_data()
model = MatrixFactorization(num_users, num_books, embedding_dim=50).to(
    device
)  # Embedding dim = hyperparameter, also move device to GPU if there is one
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 10

# Make sure models folder exists before we save checkpoints
os.makedirs("models", exist_ok=True)

"""Training Loop

1. wipe out old gradients
2. forward pass
3. measure error
4. compute gradients
5. nudge numbers
"""

print("Starting Training Now!")
best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} / {num_epochs}")

    for users, books, actual_ratings in progress:
        users = users.to(device)
        books = books.to(device)
        actual_ratings = actual_ratings.to(device)

        optimizer.zero_grad()
        predicted_ratings = model(users, books)
        loss = loss_fn(predicted_ratings, actual_ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.4f}")
    # save pytorch checkpoint every epoch in this path
    torch.save(model.state_dict(), "models/mf_latest.pt")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "models/mf_baseline.pt")
        print(f" new best (loss {best_loss:.4f}) saved to models/mf_baseline.pt")

    print(
        f"Training complete. Best model saved at models/mf_baseline.pt (loss {best_loss:.4f})"
    )
