import torch
import torch.nn as nn

# ==========================================
# 1. ARCHITECTURE (Must match exactly!)
# ==========================================
INPUT_SIZE = 18 
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3
SEQ_LENGTH = 30

class PseudoHapticLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PseudoHapticLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # We explicitly map the hidden states to the CPU for the ONNX export
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ==========================================
# 2. LOAD THE TRAINED WEIGHTS
# ==========================================
print("Loading PyTorch model...")
model = PseudoHapticLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# We load it directly to the CPU. Unity Sentis handles its own GPU routing later.
model.load_state_dict(torch.load("vr_haptic_lstm.pth", map_location=torch.device('cpu')))

# CRITICAL: Put the model in evaluation mode. 
# This locks the weights and tells PyTorch we are done training.
model.eval()

# ==========================================
# 3. EXPORT TO ONNX
# ==========================================
# We create a "dummy" input tensor that exactly matches the shape Unity will send.
# Shape: (Batch Size, Sequence Length, Features) -> (1 player, 30 frames, 18 numbers)
dummy_input = torch.randn(1, SEQ_LENGTH, INPUT_SIZE)

onnx_filename = "vr_haptic_lstm.onnx"

print("Compiling to ONNX format...")
torch.onnx.export(
    model,                      # The loaded model
    dummy_input,                # The dummy shape
    onnx_filename,              # Where to save it
    export_params=True,         # Store the trained weights inside the file
    opset_version=11,           # Opset 11 is highly stable for Unity Sentis
    do_constant_folding=True,   # Optimizes the math for faster game frame rates
    input_names=['input_sequence'],    # What C# will call the input
    output_names=['predicted_vel']     # What C# will call the output
)

print(f"Success! Your model is game-ready: {onnx_filename}")