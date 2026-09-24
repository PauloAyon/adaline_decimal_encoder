import matplotlib.pyplot as plt


def plot_mse_history(results):
    # Plot the MSE stored during each experiment.
    if not results:
        print("There are no experiments to visualize.")
        return

    plt.figure(figsize=(10, 6))

    for name, adaline in results:
        mse_values = [
            step["mse"] for step in adaline.history
            if step["mse"] is not None
        ]
        epochs = range(1, len(mse_values) + 1)
        plt.plot(epochs, mse_values, label=name)

    plt.title("ADALINE Mean Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
