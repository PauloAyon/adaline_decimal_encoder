from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def plot_mse_history(results, X, y):
    # Plot the MSE stored during each experiment and the learned encoding.
    if not results:
        print("There are no experiments to visualize.")
        return

    fig, (ax_mse, ax_enc) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left subplot: MSE convergence curve ---
    for name, adaline in results:
        mse_values = [
            step["mse"] for step in adaline.history
            if step["mse"] is not None
        ]
        epochs = range(1, len(mse_values) + 1)
        ax_mse.plot(epochs, mse_values, marker="o", label=name)

    ax_mse.set_title("ADALINE Convergence (MSE)")
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("Mean Squared Error (MSE)")
    ax_mse.grid(True)
    ax_mse.legend()

    # --- Right subplot: binary-to-decimal encoding learned ---
    x_labels = [''.join(str(int(b)) for b in row) for row in X]
    x_pos = np.arange(len(X))
    bar_width = 0.25

    ax_enc.bar(
        x_pos - bar_width,
        y,
        width=bar_width,
        label="Target (decimal)",
        color="tab:blue",
    )
    for index, (name, adaline) in enumerate(results):
        ax_enc.bar(
            x_pos + index * bar_width,
            adaline.predict(X),
            width=bar_width,
            label=f"ADALINE ({name})",
            color=f"C{index + 1}",
        )

    ax_enc.set_xticks(x_pos)
    ax_enc.set_xticklabels(x_labels)
    ax_enc.set_title("Binary to Decimal Encoding")
    ax_enc.set_xlabel("Binary input")
    ax_enc.set_ylabel("Decimal value")
    ax_enc.legend()
    ax_enc.grid(True, axis="y")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "convergence.png", dpi=150)
    plt.show()