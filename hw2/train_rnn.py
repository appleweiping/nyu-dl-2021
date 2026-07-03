"""Train + verify both HW2 GRU echo models on CPU, writing evidence to
``results/hw2/``.

Targets from the assignment:
    * fixed-delay error rate < 1%  (accuracy > 0.99)
    * variable-delay accuracy > 0.99 for delays in [0, 8]
    * each model trains within ~10 minutes
"""
import os
import time

import torch

torch.set_num_threads(3)

from rnn_echo import (  # noqa: E402
    N,
    EchoDataset,
    GRUMemory,
    VariableDelayEchoDataset,
    VariableDelayGRUMemory,
    idx_to_onehot,
    test_model,
    test_variable_delay_model,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results", "hw2"))
os.makedirs(RESULTS, exist_ok=True)
DEVICE = torch.device("cpu")


def train_fixed(delay=4, hidden=64, epochs=3, seq_length=15, size=40000, bs=64):
    torch.manual_seed(0)
    ds = EchoDataset(delay=delay, seq_length=seq_length, size=size)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs)
    model = GRUMemory(hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    last = 0.0
    for ep in range(epochs):
        for seq, target in dl:
            inp = idx_to_onehot(seq).to(DEVICE)
            target = target.to(DEVICE)
            logits = model(inp)
            loss = lossf(logits.reshape(-1, N + 1), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        print(f"  [fixed] epoch {ep + 1}/{epochs} loss={last:.4f} "
              f"t={time.time() - t0:.1f}s")
    acc = test_model(model, DEVICE, delay, seq_length)
    return model, acc, time.time() - t0


def train_variable(max_delay=8, hidden=128, epochs=12, seq_length=20, size=40000, bs=64):
    torch.manual_seed(0)
    ds = VariableDelayEchoDataset(max_delay=max_delay, seq_length=seq_length, size=size)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs)
    model = VariableDelayGRUMemory(hidden, max_delay, DEVICE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    last = 0.0
    for ep in range(epochs):
        for seq, delay, target in dl:
            inp = idx_to_onehot(seq).to(DEVICE)
            logits = model(inp, delay.to(DEVICE))
            loss = lossf(logits.reshape(-1, N + 1), target.reshape(-1).to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        sched.step()
        print(f"  [var]   epoch {ep + 1}/{epochs} loss={last:.4f} "
              f"t={time.time() - t0:.1f}s")
    acc = test_variable_delay_model(model, seq_length)
    return model, acc, time.time() - t0


def main():
    print("Training fixed-delay GRU (delay=4) ...")
    fixed_model, fixed_acc, fixed_t = train_fixed()
    # The task alphabet is a-z only (0 is the blank the model emits during the delay).
    demo_in = "helloworldthisisanechotest"
    fixed_out = fixed_model.test_run(demo_in, DEVICE)

    print("Training variable-delay GRU (delays 0..8) ...")
    var_model, var_acc, var_t = train_variable()
    var_out3 = var_model.test_run(demo_in, 3)
    var_out6 = var_model.test_run(demo_in, 6)

    report = os.path.join(RESULTS, "hw2_rnn_results.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("HW2 - RNN echo task (GRU), CPU / 3 threads\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Fixed-delay model (delay=4):\n")
        fh.write(f"  accuracy = {fixed_acc:.4f}  (target > 0.99)  "
                 f"{'PASS' if fixed_acc > 0.99 else 'FAIL'}\n")
        fh.write(f"  train time = {fixed_t:.1f}s\n")
        fh.write(f"  demo in : '{demo_in}'\n")
        fh.write(f"  demo out: '{fixed_out}'  (shifted right by 4)\n\n")
        fh.write("Variable-delay model (delays 0..8):\n")
        fh.write(f"  accuracy = {var_acc:.4f}  (target > 0.99)  "
                 f"{'PASS' if var_acc > 0.99 else 'FAIL'}\n")
        fh.write(f"  train time = {var_t:.1f}s\n")
        fh.write(f"  demo in     : '{demo_in}'\n")
        fh.write(f"  demo out d=3: '{var_out3}'\n")
        fh.write(f"  demo out d=6: '{var_out6}'\n")

    print("\n" + open(report, encoding="utf-8").read())


if __name__ == "__main__":
    main()
