from PointSet import PointSet
from Tree import Tree
from read_write import load_data
import evaluation
import argparse
import numpy as np

def build_tree(file_to_study, tree_size_proportion, is_fudyadt=False, **tree_params):
    """Build and evaluate a tree model, optionally using the FuDyADT method."""
    
    # Load data
    features, labels, types = load_data(file_to_study)
    
    # Determine training set size
    training_nb = int(len(features) * tree_size_proportion)
    
    # Initialize the tree with the training data
    current_tree = Tree(features[:training_nb], labels[:training_nb], types, **tree_params)
    
    # Prepare expected and actual results for evaluation
    expected_results = labels[training_nb:]
    actual_results = []
    
    # Apply FuDyADT if requested
    if is_fudyadt:
        for i, (point_features, point_label) in enumerate(zip(features[training_nb:], labels[training_nb:])):
            # Decide on the current point and add the result
            actual_results.append(current_tree.decide(point_features))
            
            # Add the point to the tree and remove an old point to maintain the size
            current_tree.add_training_point(point_features, point_label)
            del_point_feat, del_point_lab = features[i], labels[i]
            current_tree.del_training_point(del_point_feat, del_point_lab)
    else:
        # Simple decision without dynamic updating
        for point_features in features[training_nb:]:
            actual_results.append(current_tree.decide(point_features))
    
    # Print the tree if needed (optional)
    current_tree.print_tree(current_tree)
    
    # Calculate and return the F1 score
    return evaluation.F1_score(expected_results, actual_results)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Command-line tool with optional arguments")
    parser.add_argument("file_path", type=str, help="Path to the file")
    parser.add_argument("-hgt", "--height", type=int, default=5, help="Height (optional, default=5)")
    parser.add_argument("-msp", "--min_split_points", type=int, default=3, help="Min split points (optional, default=3)")
    parser.add_argument("-tsp", "--tree_size_proportion", type=float, default=0.8, help="Tree size proportion (optional, default=0.8)")
    parser.add_argument("-fudyadt", action="store_true", help="Use the FuDyADT method (optional)")

    args = parser.parse_args()

    # Execute the appropriate tree-building method
    f1_score = build_tree(
        file_to_study=args.file_path,
        tree_size_proportion=args.tree_size_proportion,
        is_fudyadt=args.fudyadt,
        h=args.height,
        min_split_points=args.min_split_points
    )
    
    print(f"F1 score: {f1_score}")