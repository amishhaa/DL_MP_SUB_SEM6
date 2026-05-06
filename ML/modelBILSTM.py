import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X = np.load("../dataset/X.npy")
y = np.load("../dataset/y.npy")

print("Loaded:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(LogDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(LogDataset(X_test, y_test), batch_size=64)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # hidden_dim * 2 because bidirectional
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        
        _, (h, _) = self.lstm(x)
        
        # forward + backward states
        h_forward = h[-2]
        h_backward = h[-1]
        h = torch.cat((h_forward, h_backward), dim=1)
        
        return self.fc(h)

vocab_size = int(X.max()) + 1
model = BiLSTMClassifier(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining BiLSTM...\n")

for epoch in range(5):
    total_loss = 0
    
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model.eval()
y_pred = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1)
        y_pred.extend(preds.numpy())

print("BiLSTM Results")

print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


np.save("../report/y_pred_bilstm.npy", np.array(y_pred))