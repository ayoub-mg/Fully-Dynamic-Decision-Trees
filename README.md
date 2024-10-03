# Fully-Dynamic Approximate Decision Trees

This personal project implements the algorithms presented in **"[Fully-Dynamic Approximate Decision Trees With Worst-Case Update Time Guarantees](https://arxiv.org/pdf/2302.03994)"** (Bressan & Sozio, 2023). The core contribution of this repository is efficiently handling dynamic data, meaning that new training points can be added or removed without the need to fully rebuild the tree. This is achieved while guaranteeing worst-case time complexity for updates.

The implementation focuses on ensuring that the tree can adapt to incoming data by updating its structure dynamically. When new points are added or removed, the tree adjusts itself based on predefined thresholds, while maintaining an efficient performance, even under worst-case scenarios. This makes it particularly useful for real-time applications where the data is not static, and the model needs to evolve quickly with minimal overhead.

The **Fully-Dynamic Approximate Decision Tree (FuDyADT)** method enables the tree to efficiently incorporate new points into its training set, adjust the structure of the tree dynamically, and remove old points, all while ensuring that the decision-making process remains optimal and the tree's performance does not degrade over time.

**Author** : MARZOUG Ayoub

## Project Structure

The repository is organized as follows :

.
├── main.py                # Entry point for running the decision tree algorithm
├── Tree.py                # Core decision tree implementation
├── PointSet.py            # Handles the set of points (features, labels, types)
├── evaluation.py          # Provides evaluation functions, _e.g._ F1 score, Precision, Recall
├── read_write.py          # Functions to read and load CSV data files
└── data/                  # Directory to place CSV data files (optional)

## Usage

### Installation 

To install and run the project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/yourusername/fully-dynamic-decision-trees.git
cd fully-dynamic-decision-trees
```
2. Place CSV file with training and testing data in the `data/` directory (or another directory of your choice).

### Running the Decision Tree Algorithm

The [main.py](./main.py) script allows to build a decision tree from a CSV file, granted the following structure :

+ **First column** : Labels (binary, _e.g._, 0 or 1).
+ **Remaining columns** : Feature values (numeric or categorical).

Example command to run the decision tree algorithm :

```
python main.py path/to/your/data.csv
```

### Optional Arguments

+ `--height` or `-hgt` : Set the maximum height of the tree. Default is 5.
    ```
    python main.py path/to/your/data.csv --height 6
    ```
+ `--min_split_points` or `-msp`: Set the minimum number of points required to split a node. Default is 3.
    ```
    python main.py path/to/your/data.csv --min_split_points 4
    ```

+ `--tree_size_proportion` or `-tsp` : Set the proportion of data used for training the tree. Default is 0.8 (80% training, 20% testing).
    ```
    python main.py path/to/your/data.csv --tree_size_proportion 0.7
    ```
    
+ `--fudyadt` : Use the Fully-Dynamic Approximate Decision Tree (FuDyADT) method, which updates the tree with new training points dynamically as new data is encountered.
    ```
    python main.py path/to/your/data.csv --fudyadt
    ```

### Example Usage

To run a standard decision tree with default parameters:

```
python main.py data/example_data.csv
```

To use the Fully-Dynamic Approximate Decision Tree method with a custom tree size proportion:

```
python main.py data/example_data.csv --tree_size_proportion 0.5 --fudyadt
```
### Output

The program will print the structure of the built decision tree and output the F1 score of the predictions on the test set. The tree structure and predictions will be based on the training and test split determined by the `tree_size_proportion` parameter.
