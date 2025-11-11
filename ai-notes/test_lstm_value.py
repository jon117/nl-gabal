#!/usr/bin/env python3
"""
Test: What does LSTM give us?

Compare LSTM vs FFN at level1_fast on a sequence task.
"""

import torch
import torch.nn as nn

# Test 1: Sequential dependency task
print("="*70)
print("TEST: Can the model learn sequential dependencies?")
print("="*70)

# Create a simple sequence task: predict next character
# Pattern: "abcabc..." (repeating)
sequence = "abcabcabcabc" * 10
vocab = {'a': 0, 'b': 1, 'c': 2}
tokens = torch.tensor([vocab[c] for c in sequence])

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size=3, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.output(out)
        return logits

# FFN model (no sequential processing)
class FFNModel(nn.Module):
    def __init__(self, vocab_size=3, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        emb = self.embedding(x)
        out = self.ffn(emb)
        logits = self.output(out)
        return logits

# Train both models
def train_model(model, tokens, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Input: all but last token
        # Target: all but first token (predict next)
        x = tokens[:-1].unsqueeze(0)
        y = tokens[1:]
        
        logits = model(x)
        loss = criterion(logits.squeeze(0), y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            # Check accuracy
            preds = logits.squeeze(0).argmax(dim=-1)
            acc = (preds == y).float().mean().item()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2%}")
    
    return model

print("\n1. Training LSTM model...")
lstm_model = LSTMModel()
lstm_model = train_model(lstm_model, tokens)

print("\n2. Training FFN model...")
ffn_model = FFNModel()
ffn_model = train_model(ffn_model, tokens)

# Test: Can they predict the pattern?
print("\n" + "="*70)
print("RESULTS: Pattern Prediction")
print("="*70)

test_input = torch.tensor([[vocab['a'], vocab['b']]])  # "ab" -> should predict "c"

with torch.no_grad():
    lstm_pred = lstm_model(test_input)[0, -1].argmax().item()
    ffn_pred = ffn_model(test_input)[0, -1].argmax().item()

inv_vocab = {v: k for k, v in vocab.items()}

print(f"\nInput sequence: 'ab'")
print(f"Expected next: 'c' (token {vocab['c']})")
print(f"LSTM predicts: '{inv_vocab[lstm_pred]}' (token {lstm_pred})")
print(f"FFN predicts:  '{inv_vocab[ffn_pred]}' (token {ffn_pred})")

if lstm_pred == vocab['c']:
    print("\n✅ LSTM successfully learned the sequential pattern!")
else:
    print("\n❌ LSTM failed to learn the pattern")

if ffn_pred == vocab['c']:
    print("✅ FFN successfully learned the pattern")
else:
    print("❌ FFN failed to learn the sequential pattern")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
LSTM advantage: Can model temporal dependencies through hidden states
FFN limitation: Each position is processed independently

For our Nested Learning:
- Level 1 (LSTM): Handles sequential context, short-term memory
- Level 2-3 (FFN): Fine for integration and abstraction layers

But we're NOT using LSTM to its full potential because we reset states!
Next step: Implement persistent LSTM states (Tier 3)
""")
