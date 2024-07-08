import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pdb

TICKER = "SMCI"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, ticker, seq_len, transform=None):
        self.seq_len = seq_len
        self.transform = transform

        data = yf.download(ticker, period="1y", interval="1h")
        self.data = data['Close'].values
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)  # Standardize

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features=1, n_head=4, num_layers=4, d_model=128, dim_feedforward=512, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(n_features, d_model)
        self.embedding2 = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, n_head, num_layers, num_layers, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, n_features)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.embedding2(tgt)
        output = self.transformer(src, tgt)
        output = self.fc_out(output[-1])
        return output

    def predict(self, src, future_steps):
        self.eval()
        predictions = []
        src = src.to(DEVICE)
        src = src.unsqueeze(2)  # Add feature dimension
        with torch.no_grad():
            for _ in range(future_steps):
                tgt = src[-1]
                output = self.forward(src, tgt.unsqueeze(0))
                predictions.append(output.cpu().numpy())
                src = torch.cat((src, output.unsqueeze(0)), dim=0)[1:]
        return np.array(predictions)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = x.unsqueeze(2)  # Add feature dimension
        y = y.unsqueeze(1)  # Add feature dimension

        optimizer.zero_grad()
        output = model(x.transpose(0, 1), y.unsqueeze(0))
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}, Training Loss: {epoch_loss:.4f}')
    return epoch_loss

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.unsqueeze(2)  # Add feature dimension
            y = y.unsqueeze(1)  # Add feature dimension
            output = model(x.transpose(0, 1), y.unsqueeze(0))
            loss = criterion(output, y)
            running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(val_loader.dataset)
    print(f'Validation Loss: {epoch_loss:.4f}')
    return epoch_loss


model = TimeSeriesTransformer().to(DEVICE)
SEQ_LEN = 50
dataset = StockDataset(TICKER, SEQ_LEN)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, criterion, optimizer, epoch)
    val_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    test_data = dataset.data[-SEQ_LEN:].reshape(-1, 1)  # Use the last SEQ_LEN data points for testing
    test_data = torch.tensor(test_data, dtype=torch.float32).to(DEVICE)  # Add batch and feature dimensions
    future_steps = 512  # Number of future steps to predict
    predicted = model.predict(test_data, future_steps)

# Plot predictions
plt.plot(np.arange(len(dataset.data)), dataset.data, label='Actual Data')
plt.plot(np.arange(len(dataset.data), len(dataset.data) + future_steps), predicted.squeeze(), label='Predicted Data')
plt.legend()
plt.show()