
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------
# 1. Reproducibility and device
# --------------------------------------------------

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 2. Create a synthetic classification dataset
# --------------------------------------------------

NUM_SAMPLES = 4000
INPUT_DIM = 16
NUM_CLASSES = 4

X = torch.randn(NUM_SAMPLES, INPUT_DIM)

# Generate labels using a teacher model
teacher = nn.Linear(INPUT_DIM, NUM_CLASSES)

with torch.no_grad():
    y = teacher(X).argmax(dim=1)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

X_test = torch.randn(1000, INPUT_DIM)

with torch.no_grad():
    y_test = teacher(X_test).argmax(dim=1)

X = X.to(device)
y = y.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


# --------------------------------------------------
# 3. Define the model
# --------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUT_DIM, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --------------------------------------------------
# 4. Training and evaluation functions
# --------------------------------------------------

def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean()

    return accuracy.item()


def train_model(model, epochs, learning_rate):
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            loss.backward()
            optimizer.step()

    return model


# --------------------------------------------------
# 5. Symmetric INT4 fake quantization
# --------------------------------------------------

def fake_quantize_int4(weight):
    """
    Simulate symmetric INT4 quantization.

    Signed symmetric range: -7 to +7.
    This function returns a floating-point
    quantize-dequantize approximation.
    """

    qmax = 7

    # Use a scale based on the maximum absolute weight.
    scale = weight.detach().abs().max() / qmax

    # Protect against division by zero.
    scale = scale.clamp(min=1e-8)

    # Quantize to integer levels.
    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    # Dequantize back to floating point.
    dequantized = q * scale

    return dequantized


# --------------------------------------------------
# 6. Train FP32 baseline
# --------------------------------------------------

fp32_model = MLP().to(device)

fp32_model = train_model(
    fp32_model,
    epochs=100,
    learning_rate=0.01
)

fp32_accuracy = evaluate(
    fp32_model, X_test, y_test
)

print("\nFP32 accuracy:", fp32_accuracy)


# --------------------------------------------------
# 7. PTQ: Quantize after training
# --------------------------------------------------

ptq_model = copy.deepcopy(fp32_model)

ptq_model.eval()

with torch.no_grad():
    for parameter in ptq_model.parameters():
        parameter.copy_(
            fake_quantize_int4(parameter)
        )

ptq_accuracy = evaluate(
    ptq_model, X_test, y_test
)

print("PTQ INT4 accuracy:", ptq_accuracy)


# --------------------------------------------------
# 8. QAT: Fine-tune with fake quantization
# --------------------------------------------------

class QATLinear(nn.Module):
    def __init__(self, original_linear):
        super().__init__()

        self.weight = nn.Parameter(
            original_linear.weight.detach().clone()
        )

        self.bias = nn.Parameter(
            original_linear.bias.detach().clone()
        )

    def forward(self, x):

        # Forward: use quantized-dequantized weights.
        quantized_weight = fake_quantize_int4(
            self.weight
        )

        # STE:
        # Forward value is quantized_weight.
        # Backward gradient passes to self.weight.
        weight_ste = (
            self.weight
            + (quantized_weight - self.weight).detach()
        )

        return F.linear(
            x,
            weight_ste,
            self.bias
        )


class QATMLP(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()

        self.fc1 = QATLinear(pretrained_model.fc1)
        self.fc2 = QATLinear(pretrained_model.fc2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


qat_model = QATMLP(fp32_model).to(device)

qat_model = train_model(
    qat_model,
    epochs=50,
    learning_rate=0.001
)

qat_accuracy = evaluate(
    qat_model, X_test, y_test
)

print("QAT INT4 accuracy:", qat_accuracy)


# --------------------------------------------------
# 9. Compare results
# --------------------------------------------------

print("\n" + "=" * 45)
print("QUANTIZATION COMPARISON")
print("=" * 45)

print(f"FP32 baseline : {fp32_accuracy:.4f}")
print(f"PTQ INT4      : {ptq_accuracy:.4f}")
print(f"QAT INT4      : {qat_accuracy:.4f}")

print("\nAccuracy changes relative to FP32:")
print(f"PTQ change: {(ptq_accuracy - fp32_accuracy) * 100:.2f} percentage points")
print(f"QAT change: {(qat_accuracy - fp32_accuracy) * 100:.2f} percentage points")


# --------------------------------------------------
# 10. Verify STE gradients
# --------------------------------------------------

test_weight = torch.tensor(
    [0.7234, -0.4321, 0.1234],
    requires_grad=True,
    device=device
)

quantized_weight = fake_quantize_int4(test_weight)

weight_ste = (
    test_weight
    + (quantized_weight - test_weight).detach()
)

loss = weight_ste.sum()
loss.backward()

print("\nSTE gradient test")
print("Original weights:", test_weight.detach())
print("Quantized values:", quantized_weight.detach())
print("Gradients:", test_weight.grad)

assert torch.allclose(
    test_weight.grad,
    torch.ones_like(test_weight)
)

print("\nAll gradient checks passed.")
