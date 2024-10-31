import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import os
# from utils import read_inputs, count_letters, print_line, score
from .dataset import CharCountDataset

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.layer_norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, d_model = x.size()
        
        # Transform and split into heads
        Q = self.query(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.fc_out(attn_output)
        
        return output

# Custom Transformer Model
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, max_len):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_enc = nn.Parameter(torch.randn(max_len, d_model))  # Learned positional encodings

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)  # Output layer for character counts

    def forward(self, x):
        batch_size, seq_length = x.size()
        embedded = self.embedding(x) + self.positional_enc[:seq_length, :]
        
        # Apply transformer layers
        for layer in self.layers:
            embedded = layer(embedded)
        
        logits = self.fc_out(embedded)
        return logits
    
    def generate(self, x):
        """
        Generates character count predictions for an input sequence.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length).
        
        Returns:
            logits (Tensor): Raw logits of shape (batch_size, seq_length, vocab_size).
            predictions (Tensor): Predicted character counts (using argmax) or probabilities, shape (batch_size, seq_length, vocab_size).
        """
        # Forward pass to get logits
        logits = self.forward(x.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get the predicted counts (e.g., using argmax)
        predictions = probabilities.argmax(dim=-1)
        
        return logits, predictions

# Initialize dataset, model, optimizer, and criterion
def train_classifier(texts, epochs=3, batch_size=32, max_len=20, d_model=128, n_heads=4, num_layers=2, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset and DataLoader
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab = {char: idx for idx, char in enumerate(vocabs)}
    dataset = CharCountDataset(texts, vocab=vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, and loss function
    model = CustomTransformer(dataset.vocab_size, d_model, n_heads, num_layers, max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)  # Shape: [batch_size, max_len, vocab_size]
            outputs = outputs.view(-1, dataset.vocab_size)  # Reshape for CrossEntropyLoss
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

if __name__ == "__main__":
  texts = read_inputs(
        os.path.join("data", "train.txt")
  )
  model = train_classifier(texts)

  test_inputs = read_inputs(os.path.join("data", "test.txt"))
  test_dataset = CharCountDataset(texts=test_inputs)
  model.eval()
  
  golds = []
  predictions = []
  for text, gold in test_dataset:
        logits, prediction = model.generate(text)
        predictions.append(prediction.cpu().numpy().flatten())
        golds.append(gold.numpy())

  golds = np.stack(golds)
  predictions = np.stack(predictions)

  # Print the first five inputs, golds, and predictions for analysis
  for i in range(5):
        print(f"Input {i+1}: {test_inputs[i]}")
        print(f"Gold {i+1}: {count_letters(test_inputs[i]).tolist()}")
        print(f"Pred {i+1}: {predictions[i].tolist()}")
        print_line()

  print(f"Test Accuracy: {100.0 * score(golds, predictions):.2f}%")
  print_line()
