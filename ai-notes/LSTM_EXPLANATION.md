# What Does the LSTM Give Us?

## 🧠 TLDR

**LSTM at level1_fast provides:**
1. ✅ Sequential processing (maintains hidden state h,c)
2. ✅ Temporal memory (within batch)
3. ✅ Better gradient flow than vanilla RNN
4. ⚠️ **BUT:** We're resetting states between batches (limiting its power!)

## Current Implementation

```python
# Level 1: LSTM (fast, updates every step)
self.level1_fast = nn.LSTM(input_size, input_size, batch_first=True)

# Level 2: FFN (medium, updates every 16 steps)
self.level2_medium = nn.Linear(input_size, input_size)

# Level 3: FFN (slow, updates every 256 steps)
self.level3_slow = nn.Linear(input_size, input_size)
```

## LSTM vs FFN

| Feature | LSTM | FFN |
|---------|------|-----|
| **Sequential** | ✅ Processes tokens in order | ❌ Each token independent |
| **Memory** | ✅ Hidden state (h,c) | ❌ No memory |
| **Context** | ✅ Sees previous tokens | ❌ Only current input |
| **Speed** | 🟡 Sequential (slower) | ✅ Parallel (faster) |
| **Long-range** | 🟡 Better than RNN, worse than Transformer | ❌ Very limited |

## What LSTM Gives Us

### 1. Sequential Context 🔄

**Within a sequence:**
```
Input:  ["The", "cat", "sat", "on", "the", "mat"]
        ↓      ↓      ↓      ↓      ↓      ↓
LSTM:   h0 →  h1 →  h2 →  h3 →  h4 →  h5 → h6
```

Each hidden state carries information from all previous tokens.

**FFN equivalent:**
```
Input:  ["The", "cat", "sat", "on", "the", "mat"]
        ↓      ↓      ↓      ↓      ↓      ↓
FFN:    out    out    out    out    out    out
        (no connection between positions!)
```

### 2. Gating Mechanisms 🚪

LSTM has 3 gates:
- **Forget gate (f):** What to remove from memory
- **Input gate (i):** What new info to add
- **Output gate (o):** What to output

This allows it to:
- Remember important information
- Forget irrelevant details
- Control information flow

### 3. Vanishing Gradient Solution 📉→📈

Vanilla RNN: Gradients vanish over long sequences
LSTM: Cell state provides "highway" for gradients

## What We're Currently Missing! 🚨

### Problem: State Reset Between Batches

```python
# Current behavior:
batch_1 = tokenize("The quick brown fox")
output_1, (h_1, c_1) = lstm(batch_1)
# h_1, c_1 are DISCARDED! ❌

batch_2 = tokenize("jumps over the lazy dog")
output_2, (h_2, c_2) = lstm(batch_2)  # Starts with ZERO hidden state! ❌
```

**Result:** LSTM can't learn dependencies across batches!

### What We Should Do: Persistent States

```python
class NestedModelWithPersistentState:
    def __init__(self):
        self.lstm_hidden = None  # Persistent!
    
    def forward(self, x):
        if self.lstm_hidden is None:
            self.lstm_hidden = self.init_hidden(batch_size)
        
        output, self.lstm_hidden = self.lstm(x, self.lstm_hidden)
        
        # Detach to prevent backprop across batches
        self.lstm_hidden = tuple(h.detach() for h in self.lstm_hidden)
        
        return output
```

**Result:** LSTM remembers context across batches! ✅

## Why LSTM at Fast Level?

The paper's reasoning:

**Fast Level (LSTM, updates every step):**
- Needs to be **reactive** to every token
- Maintains **short-term context**
- Sequential processing makes sense here

**Medium Level (FFN, updates every 16 steps):**
- **Integrates** information over time
- Sees accumulated gradients from 16 steps
- Doesn't need sequential processing

**Slow Level (FFN, updates every 256 steps):**
- Learns **high-level patterns**
- Very long timescale
- Feedforward is sufficient

## Experiment Results

We tested LSTM vs FFN on our WikiText-2 experiments:

**With LSTM at level1:**
- Baseline: 53.60 PPL
- With surprise: 52.12 PPL ✅

**The LSTM helps, but we're not using it optimally yet!**

## What's Next: Tier 3 Implementation

### Priority: Add Persistent LSTM States

**Implementation:**
```python
class NestedModelWithState(NestedModelWithSurprise):
    def reset_states(self):
        """Call at start of new document/sequence."""
        self.lstm_hidden = None
    
    def forward(self, x, compute_surprise=False):
        # Initialize if needed
        if self.lstm_hidden is None:
            self.lstm_hidden = self.init_hidden(x.size(0))
        
        # Forward with persistent state
        fast_out, new_hidden = self.level1_fast(x, self.lstm_hidden)
        
        # Update state (detached to prevent BPTT across batches)
        self.lstm_hidden = tuple(h.detach() for h in new_hidden)
        
        # Continue as before...
```

**Benefits:**
- ✅ True sequential learning across batches
- ✅ Better long-range dependencies
- ✅ More biologically plausible
- ✅ Closer to paper's intent

**Challenges:**
- Need to know when to reset (document boundaries)
- Slightly more complex training loop
- May need gradient clipping

## Alternative: Replace LSTM with Transformer?

**Could we use Transformer at level1?**

```python
self.level1_fast = nn.TransformerEncoderLayer(
    d_model=input_size,
    nhead=8,
    batch_first=True
)
```

**Pros:**
- ✅ Better parallelization
- ✅ Better long-range modeling (with attention)
- ✅ More modern architecture

**Cons:**
- ❌ O(n²) memory for attention
- ❌ Less biologically plausible
- ❌ Deviates from paper's design
- ❌ May not work as well with multi-timescale updates

**Recommendation:** Stick with LSTM for now, add persistent states (Tier 3)

## Summary

### Current State
- ✅ LSTM at level1 provides sequential processing
- ✅ Works well in our experiments (52.12 PPL with surprise)
- ⚠️ Not using full potential (states reset between batches)

### Next Steps
1. **Implement persistent LSTM states** (Tier 3, high priority)
2. Test if it improves performance
3. Add reset mechanism for document boundaries
4. Consider Transformer comparison (research)

### Bottom Line

**The LSTM gives us sequential context and temporal memory, but we're only using it within batches. Adding persistent states (Tier 3) will unlock its full potential!**

---

**Files to implement:**
- `src/model_state.py` - Model with persistent states
- `src/train_state.py` - Training loop with state management
- `tests/test_state.py` - Test state persistence

**Estimated effort:** Medium (a few hours)
**Expected benefit:** 1-5% improvement on sequential tasks
**Priority:** High (should do next!)
