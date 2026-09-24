from adaline import Adaline
from data import get_data
from visualization import plot_mse_history


# Define independent training experiments.
EXPERIMENTS = [
    (
        "Experiment 1",
        {
            "learning_rate": 0.3,
            "initial_bias": 0,
            "epochs": 10,
        },
    ),
    (
        "Experiment 2",
        {
            "learning_rate": 0.5,
            "initial_bias": 0.3,
            "epochs": 20,
        },
    )
]


def run_experiments(X, y):
    # Train one independent ADALINE for each configuration.
    results = []

    for name, configuration in EXPERIMENTS:
        adaline = Adaline(**configuration)
        adaline.train(X, y)
        results.append((name, adaline))

    return results


def print_results(results):
    # Print the final parameters and the last recorded MSE.
    for name, adaline in results:
        print(f"\n{name}")
        print(f"  w1 = {adaline.w1}")
        print(f"  w2 = {adaline.w2}")
        print(f"  w3 = {adaline.w3}")
        print(f"  bias = {adaline.bias}")
        mse_values = [
            step["mse"] for step in adaline.history
            if step["mse"] is not None
        ]
        print(f"  final MSE = {mse_values[-1]:.10f}")


def main():
    X, y = get_data()
    results = run_experiments(X, y)

    print_results(results)
    plot_mse_history(results)


if __name__ == "__main__":
    main()
