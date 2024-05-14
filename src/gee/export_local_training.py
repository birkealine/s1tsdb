""""""

# improved and simplified from geemap.ml module
from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree._classes import DecisionTreeClassifier
from tqdm import tqdm
from typing import List


def rf_to_strings(model: RandomForestClassifier, feature_names: List[str], processes: int = 8):
    # extract out the estimator trees
    estimators = np.squeeze(model.estimators_)

    # check that number of processors set to use is not more than available
    if processes >= mp.cpu_count():
        # if so, force to use only cpu count - 1
        processes = mp.cpu_count() - 1

    # run the tree extraction process in parallel
    with mp.Pool(processes) as pool:
        proc = pool.map_async(
            partial(tree_to_string, feature_names=feature_names, verbose=0),
            estimators,
        )
        trees = list(proc.get())

    return trees


def tree_to_string(estimator: DecisionTreeClassifier, feature_names: List[str], verbose: int = 0):
    """The main function to convert a sklearn decision tree to a tree string"""
    # extract out the information need to build the tree string
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature_idx = estimator.tree_.feature
    impurities = estimator.tree_.impurity
    n_samples = estimator.tree_.n_node_samples
    thresholds = estimator.tree_.threshold
    features = [feature_names[i] for i in feature_idx]
    raw_vals = np.squeeze(estimator.tree_.value)

    # ==== PROBABILITY ==== #
    # calculate fraction of samples of the same class in a leaf
    # currently only supporting binary classifications
    # check if n classes == 2 (i.e. binary classes)
    if raw_vals.shape[-1] != 2:
        raise ValueError(
            f"shape mismatch: outputs from trees = {raw_vals.shape[-1]} classes, currently probability outputs is support for binary classifications"
        )
    probas = np.around((raw_vals / np.sum(raw_vals, axis=1)[:, np.newaxis]), decimals=6)
    values = probas[:, -1]
    out_type = float

    # use iterative pre-order search to extract node depth and leaf information
    node_ids = np.zeros(shape=n_nodes, dtype=np.int64)
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        node_ids[node_id] = node_id

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    # create a table of the initial structure
    # each row is a node or leaf
    df = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_depth": node_depth,
            "is_leaf": is_leaves,
            "children_left": children_left,
            "children_right": children_right,
            "value": values,
            "criterion": impurities,
            "n_samples": n_samples,
            "threshold": thresholds,
            "feature_name": features,
            "sign": ["<="] * n_nodes,
        },
        dtype="object",
    )

    # the table representation does not have lef vs right node structure
    # so we need to add in right nodes in the correct location
    # we do this by first calculating which nodes are right and then insert them at the correct index

    # ====== IMPROVED FROM geemap.ml ====== #
    # Precompute the rows to be inserted
    inserts = []
    for row in df.itertuples():
        child_r = row.children_right
        if child_r > row.Index:
            ordered_row = np.array(row)
            ordered_row[-1] = ">"
            # Store the row and the index where it should be inserted
            inserts.append((child_r, ordered_row[1:]))  # drop index value

    # Sort the rows by the insertion index
    inserts.sort(key=lambda x: x[0])

    # Create a new array to hold the combined data
    new_rows = []
    last_idx = 0
    for idx, row in inserts:
        # Add the existing rows up to the current insertion index
        new_rows.extend(df.values[last_idx:idx])
        # Add the new row
        new_rows.append(row)
        last_idx = idx

    # Add any remaining rows from the original array
    new_rows.extend(df.values[last_idx:])

    # Convert the list of rows back to a numpy array
    table_values = np.array(new_rows)
    # ===================================== #

    # make the ordered table array into a dataframe
    # note: df is dtype "object", need to cast later on
    ordered_df = pd.DataFrame(table_values, columns=df.columns)

    ordered_df = ordered_df.astype(
        {
            "node_id": int,
            "node_depth": int,
            "is_leaf": bool,
            "children_left": int,
            "children_right": int,
            "value": float,
            "criterion": float,
            "n_samples": int,
            "threshold": float,
            "feature_name": str,
            "sign": str,
        }
    )

    if verbose:
        print("Dataframe ready for tree string conversion")

    max_depth = np.max(ordered_df.node_depth.astype(int))
    tree_str = f"1) root {n_samples[0]} 9999 9999 ({impurities.sum()})\n"
    previous_depth = -1
    cnts = []
    # loop through the nodes and calculate the node number and values per node
    if verbose:
        pbar = tqdm(total=len(ordered_df))
    for i, row in enumerate(ordered_df.itertuples()):
        node_depth = int(row.node_depth)
        left = int(row.children_left)
        right = int(row.children_right)
        if left != right:
            if row.Index == 0:
                cnt = 2
            elif previous_depth > node_depth:
                depths = ordered_df.node_depth.values[: row.Index]
                idx = np.where(depths == node_depth)[0][-1]
                # cnt = (cnts[row.Index-1] // 2) + 1
                cnt = cnts[idx] + 1
            elif previous_depth < node_depth:
                cnt = cnts[row.Index - 1] * 2
            elif previous_depth == node_depth:
                cnt = cnts[row.Index - 1] + 1

            if node_depth == (max_depth - 1):
                value = out_type(ordered_df.iloc[row.Index + 1].value)
                samps = int(ordered_df.iloc[row.Index + 1].n_samples)
                criterion = float(ordered_df.iloc[row.Index + 1].criterion)
                tail = " *\n"
            else:
                if (
                    (bool(ordered_df.loc[ordered_df.node_id == left].iloc[0].is_leaf))
                    and (bool(int(row.Index) < int(ordered_df.loc[ordered_df.node_id == left].index[0])))
                    and (str(row.sign) == "<=")
                ):
                    rowx = ordered_df.loc[ordered_df.node_id == left].iloc[0]
                    tail = " *\n"
                    value = out_type(rowx.value)
                    samps = int(rowx.n_samples)
                    criterion = float(rowx.criterion)

                elif (
                    (bool(ordered_df.loc[ordered_df.node_id == right].iloc[0].is_leaf))
                    and (bool(int(row.Index) < int(ordered_df.loc[ordered_df.node_id == right].index[0])))
                    and (str(row.sign) == ">")
                ):
                    rowx = ordered_df.loc[ordered_df.node_id == right].iloc[0]
                    tail = " *\n"
                    value = out_type(rowx.value)
                    samps = int(rowx.n_samples)
                    criterion = float(rowx.criterion)

                else:
                    value = out_type(row.value)
                    samps = int(row.n_samples)
                    criterion = float(row.criterion)
                    tail = "\n"

            # extract out the information needed in each line
            spacing = (node_depth + 1) * "  "  # for pretty printing
            fname = str(row.feature_name)  # name of the feature (i.e. band name)
            tresh = float(row.threshold)  # threshold
            sign = str(row.sign)

            tree_str += f"{spacing}{cnt}) {fname} {sign} {tresh:.6f} {samps} {criterion:.4f} {value}{tail}"
            previous_depth = node_depth
        cnts.append(cnt)

        if i % 5000 == 0 and verbose:
            pbar.update(5000)

    print("Tree string conversion complete")

    return tree_str
