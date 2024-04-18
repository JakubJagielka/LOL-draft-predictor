import pandas as pd
import ast
import torch
import numpy as np
def convert_to_list(column):
    try:
        # Attempt to evaluate the string as a Python literal (list)
        return ast.literal_eval(column)
    except Exception as e:
        # In the case of an error, return the original string
        # Optionally print/log the error message and/or problematic data
        print(f"Error converting data to list: {e}")
        return column


def process_dataframe(data):
# Iterate over columns and apply conversion to each string column
    for i in data.columns:
        # Check if the first entity in the column is a string and tries to convert it
        if isinstance(data[i][0], str):
            data[i] = data[i].apply(convert_to_list)



    return data
def create_tensors_for_embeddings(data, device):
    data = process_dataframe(data)
    context_data = data.iloc[:, 3:]
    assert data.shape[1] == 27

    champions = torch.tensor(np.stack(data['championId'].tolist()), dtype=torch.long).to(device)
    results = torch.tensor(data['Winning Team'].tolist(), dtype=torch.float).to(device)
    results = results.reshape(869498,1)
    context = [torch.tensor(context_data[col].tolist(), dtype=torch.float).to(device) for col in context_data]
    context = torch.stack(context, dim=1)
    context = context.transpose(1,2)

    return champions, context, results

def create_tensors(data, result, device):
    data = pd.read_csv(data)
    result = pd.read_csv(result)
    assert data.shape[1] == 10
    assert result.shape[1] == 1
    d = torch.tensor(data.values, dtype=torch.long).to(device)
    d1 = torch.tensor(result.values, dtype=torch.float).to(device)

    return d, d1