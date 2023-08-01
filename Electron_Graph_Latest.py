import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from networkx.linalg.graphmatrix import adjacency_matrix
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader, Dataset, Data
import random
import csv
import glob

edge_index = []
x = []
data_list=[]
energy=[]


file_names = glob.glob('/home/4tb_Drive_1/HGCAL_HLT/CSV/*.csv')
for file_name in file_names:
    f = int(file_name.split('_')[-1].split('.')[0])
    df = pd.read_csv(file_name)
    print(f'graph going on is for electron {f}')

    if not df.empty:  # check if the DataFrame is empty
        # Split the dataset by layer number using a for loop
        layer_data = []
        energy_event = df['E'].iloc[0]
        energy.append(energy_event)
        print("Energy for event", f, "is equal to:", energy_event)

        for layer_num in range(1, 47):  # since electron travels upto 30th layer
            layer_df = df.loc[df['Layer'] == layer_num]  # filter rows by layer number
            layer_data.append(layer_df)
    else:
        print("Empty DataFrame for event", f)
        #print(layer_data)

    # Sort the DataFrame by Layer and effective ADC
    df.sort_values(['Layer', 'Eff_ADC'], inplace=True)

    # Initialize an empty graph
    graph = nx.Graph()

    # Create nodes and add them to the graph
    for index, row in df.iterrows():
        graph.add_node(index, x=row['X'], y=row['Y'], layer=row['Layer'], adc=row['Eff_ADC'])

    # Connect nodes within all layers. Nodes of each layer, connected to all nodes of that layer.
    layers = df['Layer'].unique()

    for l in layers:
        nodes_in_layer = df[df['Layer'] == l].index.tolist()
        first_node = nodes_in_layer[0]
        for node in nodes_in_layer[1:]:
            graph.add_edge(first_node, node)


    # Connect nodes between consecutive layers based on distance
    for i in range(len(layers) - 1):
        current_layer_nodes = df[df['Layer'] == layers[i]].index.tolist()
        next_layer_nodes = df[df['Layer'] == layers[i+1]].index.tolist()

        for current_node in current_layer_nodes:
            distances = []
            for next_node in next_layer_nodes:
                distance = ((df.loc[current_node, 'X'] - df.loc[next_node, 'X'])**2 + (df.loc[current_node, 'Y'] - df.loc[next_node, 'Y'])**2)**0.5
                distances.append((distance, next_node))
            distances.sort()  # Sort distances in ascending order
            for _, nearest_node in distances[:5]:
                graph.add_edge(current_node, nearest_node)
    #print("--------------------------Adjacency Matrix---------------------------")

    '''# Create a layout for the graph using Kamada-Kawai algorithm
    pos = nx.kamada_kawai_layout(graph)

    # Draw the nodes
    nx.draw_networkx_nodes(graph, pos, node_size=10)

    # Draw the edges
    nx.draw_networkx_edges(graph, pos)

    # Add labels to the nodes (optional)
    #labels = {node: str(node) for node in graph.nodes()}
    #nx.draw_networkx_labels(graph, pos, labels)

    # Display the graph
    plt.axis('off')
    plt.show()'''

    # Make adjacency matrix and convert to tensor

    vertices = set()
    for edge in graph.edges():
      vertices.update(edge)

    # Create a dictionary to map each vertex to a unique index
    index_map = {vertex: i for i, vertex in enumerate(vertices)}

    # Create the adjacency matrix
    n = len(vertices)
    adj_matrix = [[0] * n for _ in range(n)]
    for edge in graph.edges():
      i, j = index_map[edge[0]], index_map[edge[1]]
      adj_matrix[i][j] = 1
      adj_matrix[j][i] = 1


    # Print the adjacency matrix
    #for row in adj_matrix:
      # print(row)

    #create an empty list for all matrices
    #print("----------------------------Edge Index---------------------------------")

    try:
      arr = np.array(adj_matrix)
      rows, cols = np.where(arr == 1)
      adj_tensor = torch.tensor([rows, cols], dtype=torch.long)
      edge_index.append(adj_tensor)
    except (ValueError, IndexError) as e:
        print(f"Skipping file: {f}. Error: {e}")
        continue

    # Make Feature matrix and convert to tensor

    #print("----------------------------Feature Matrix------------------------------")
    # Extract the two feature columns as a 2D numpy array
    adc_data = []
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            adc_data.append(row)

    # create the dictionary with node features as ADC values
    node_features = {}
    for i, row in enumerate(adc_data):
        node_features[i] = {'adc': float(row['Eff_ADC']), 'layer': int(row['Layer']), 'X': float(row['X']), 'Y': float(row['Y'])}

    # create the feature matrix
    feature_matrix = []
    for i in range(len(adc_data)):
        features = []
        for key in node_features[i]:
            features.append(node_features[i][key])
        feature_matrix.append(features)

    #print("feature matrix: ", feature_matrix)

    #print("-----------------------------------X list---------------------------------")
    # Convert the numpy array to a PyTorch tensor
    o = torch.tensor(feature_matrix, dtype=torch.float)

    #print("Tensor feature matrix: ",o)
    x.append(o)

#inserting data into the data_list
for k in range(0, len(file_names)):
    data_list.append(Data(x=x[k], edge_index=edge_index[k], y=torch.tensor([energy[k]], dtype=torch.float)))
    print("Datalist is filled for:", file_name)

import pickle
# assume `outer_list` is the list of objects you want to save
with open('Electron_latest.pkl', 'wb') as f:
    pickle.dump(data_list, f)
