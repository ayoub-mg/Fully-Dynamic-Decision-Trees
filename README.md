# Fully-Dynamic Approximate Decision Trees

This personal project implements the algorithms presented in **"[Fully-Dynamic Approximate Decision Trees With Worst-Case Update Time Guarantees](https://arxiv.org/pdf/2302.03994)"** (Bressan & Sozio, 2023). The core idea behind this implementation is to build decision trees that can efficiently handle dynamic data, meaning that new training points can be added or removed without the need to fully rebuild the tree. This is achieved while guaranteeing worst-case time complexity for updates.

The implementation focuses on ensuring that the tree can adapt to incoming data by updating its structure dynamically. When new points are added or removed, the tree adjusts itself based on predefined thresholds, while maintaining an efficient performance, even under worst-case scenarios. This makes it particularly useful for real-time applications where the data is not static, and the model needs to evolve quickly with minimal overhead.

The **Fully-Dynamic Approximate Decision Tree (FuDyADT)** method enables the tree to efficiently incorporate new points into its training set, adjust the structure of the tree dynamically, and remove old points, all while ensuring that the decision-making process remains optimal and the tree's performance does not degrade over time.


Project Structure

graphql

├── main.py                # Entry point for running the decision tree algorithm
├── Tree.py                # Core decision tree implementation
├── PointSet.py            # Handles the set of points (features, labels, types)
├── evaluation.py          # Provides evaluation functions like F1 score
├── read_write.py          # Functions to read and load CSV data files
└── data/                  # Directory to place your CSV data files (optional)

Usage
Running the Decision Tree Algorithm

To run the decision tree algorithm, execute the main.py script with a CSV file as input. The CSV file should have the following structure:

    First column: Labels (binary, e.g., 0 or 1).
    Remaining columns: Feature values (numeric or categorical).

Basic command:

bash

python main.py path/to/your/data.csv

Optional Arguments

    --height or -hgt: Set the maximum height of the tree (default: 5).
    --min_split_points or -msp: Minimum points required to split a node (default: 3).
    --tree_size_proportion or -tsp: Proportion of data for training (default: 0.8).
    --fudyadt: Use the Fully-Dynamic Approximate Decision Tree (FuDyADT) method.

Example Commands

Run the decision tree with default parameters:

bash

python main.py data/example_data.csv

Use the Fully-Dynamic Approximate Decision Tree method:

bash

python main.py data/example_data.csv --fudyadt

Output

The script will print the decision tree structure and the F1 score of the predictions on the test set.
