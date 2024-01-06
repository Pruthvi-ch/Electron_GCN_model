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
import argparse as arg

parser = arg.ArgumentParser(description='Find Accuracy')
parser.add_argument('-s', '--start', dest='start', type=str, default='0', help="start")
parser.add_argument('-t', '--tag', dest='tag', type=str, default='', help="tag")
args = parser.parse_args()

edge_index = []
x = []
data_list=[]
energy=[]
start = int(args.start)
end = start + 1000
tag = args.tag
#fixed2 = np.loadtxt('Graph_Regular_Electron_PU_000_Nominal_60k.csv', delimiter=',')
print('/nfs/home/common/HGCAL_HLT/Output/Detection_CSV_Output/Wt_EJ/Mono/Electron/Graph_Mono_Electron_' + tag + '_Nominal.csv')
fixed2 = np.loadtxt('/nfs/home/common/HGCAL_HLT/Output/Detection_CSV_Output/Wt_EJ/Mono/Electron/Graph_Mono_Electron_' + tag + '_Nominal.csv', delimiter=',') 
print(np.shape(fixed2))

file_names = glob.glob('CSV_' + tag + '/Event_fet_*.csv')
print(len(file_names))
if end > len(file_names): end = -1
print(start, '    ', end)
for file_name in file_names[start:end]:
    f = int(file_name.split('_')[-1].split('.')[0])
    df = pd.read_csv(file_name)
    print(f'graph going on is for electron {f}')

    if not df.empty:  # check if the DataFrame is empty
        # Split the dataset by layer number using a for loop
        layer_data = []
        fixed = np.loadtxt('CSV_' + tag + '/Event_all_' + str(f) + '.csv', delimiter=',')
        print(f, "   ", file_name, "   ", fixed)
        found = False
        for jj in range(np.shape(fixed2)[0]):
            if f == int(fixed2[jj,0]):
                fixed3 = fixed2[jj, 1:]
                found = True
                break
        #print(f, fixed2[jj, 0])
        if not found: continue    
        #print(fixed3, fixed)
        fixed3[2] = fixed[2]
        print(fixed3, fixed)
        energy.append(fixed3)
        print("Energy for event", f, "is equal to:", fixed[3])

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
      adj_tensor = torch.tensor(np.array([rows, cols]), dtype=torch.int16)
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
for k in range(0, len(x)):
    print(x[k].size())
    print(edge_index[k].size())
    data_list.append(Data(x=x[k], edge_index=edge_index[k], y=torch.tensor(energy[k], dtype=torch.float)))
    print("Datalist is filled for:", file_name)

import pickle
# assume `outer_list` is the list of objects you want to save
with open('Electron_latest_' + tag + '.pkl', 'wb') as f:
    pickle.dump(data_list, f)
