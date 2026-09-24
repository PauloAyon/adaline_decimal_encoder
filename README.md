# ADALINE Decimal Encoder

This project demonstrates how an ADALINE neural network learns a relationship
between binary inputs and numerical target values.

The program trains the model using the delta rule. During training, it
calculates an output, compares it with the target value, and updates the
weights and bias to reduce the error.

After training, the program prints the final parameters and the final MSE for
each experiment. It also displays a graph that shows the error during the
training process.

## Requirements

- Python 3.9 or later
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

## Project Files

- `data.py`: Provides the input data and target values.
- `adaline.py`: Defines the ADALINE model and its training process.
- `main.py`: Runs the experiments and prints the results.
- `visualization.py`: Displays the error graph.
