import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear
from torch_geometric.data import DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import torch
from torch.nn import Linear
from torch_geometric.data import DataLoader, Dataset, Data
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import time
import argparse as arg

parser = arg.ArgumentParser(description='Find Accuracy')
parser.add_argument('-t', '--tag', dest='tag', type=str, default='', help="tag")
args = parser.parse_args()

tag = args.tag
data_list = []
num_edges = []
num_nodes = []
#file_lists = ['Electron_latest_new_18000.pkl', 'Electron_latest_new_19500.pkl', 'Electron_latest_new_21000.pkl']
import glob


#file_lists = glob.glob('Electron_latest_PU_000_60k_*.pkl')
file_lists = ['Electron_latest_' + tag + '.pkl']
#file_lists = file_lists[18000:20000]
for file_list in file_lists:
    with open(file_list, 'rb') as f:
        data_list += pickle.load(f)

test_data=data_list[:]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size=1
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
#test_loader = test_loader.to(device)

criterion = torch.nn.SmoothL1Loss()
class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc1 = Linear(hidden_channels+4, hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.fc3 = Linear(hidden_channels, 1)  # Single output node for regression
        self.reg_lambda = reg_lambda

    def forward(self, x, edge_index, y, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch)
        x = torch.cat((x,y), 1)
        #print(y)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Define the training parameters

# Create the GCN model and optimizer
# model = Net(num_features, hidden_channels)

# Load the model weights from the saved file
#model=torch.load("Electron_PU_000_new2.pt", map_location=device)
model=torch.load("Electron_PU_000_60k_2.pt", map_location=device)
print("loaded")
model.to(device)
# Set the model to evaluation mode (no gradient calculation during testing)
model.eval()
# Create data loaders for the training and testing datasets
#batch_size = 1

predictions = []
targets = []
t = []

# Train the model
fout = open('Energy_log_' + tag + '_3.txt', 'w')

with torch.no_grad():
    for batch in test_loader:
        t_start = time.time()
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        num_edges.append(edge_index.size(dim=1))
        num_nodes.append(x.size(dim=0))
        print(x.size())
        y1 = batch.y[:4].to(device)
        y1 = torch.unsqueeze(y1, 0)
        batch1 = batch.batch.to(device)
        out = model(x, edge_index, y1, batch1)
        predictions.extend(out.cpu().numpy().flatten().tolist())
        targets.extend(batch.y[4].cpu().numpy().flatten().tolist())
        out = out.squeeze()
        t_end = time.time()
        t.append(t_end - t_start)
        target = batch.y[4].float().to(device)
        loss = criterion(out, target)
        print('Actual label was->',batch.y[3],'Predicted output:', out)
        fout.write(str(targets[-1]) + '    ' + str(predictions[-1]) + '\n')

fout.close()
predictions = np.array(predictions)
targets = np.array(targets)

mse = mean_squared_error(targets, predictions)
rmse=np.sqrt(mse)

print("Best mse is = ",mse)
print("Best rmse is = ",rmse)
print("Average evaluation time = ", np.mean(np.array(t))/batch_size)

print('Average nodes = ' , np.mean(np.array(num_nodes)))
print('Average edges = ' , np.mean(np.array(num_edges)))

actual_values, predicted_values = targets,predictions
# Create a scatter plot

plt.scatter(actual_values, predicted_values, color='red', label='Predicted')
plt.scatter(actual_values, actual_values, color='blue', label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.legend()
plt.show()
plt.savefig('scatter_plot.png')

actual_labels = np.array(actual_values)
predicted_outputs = np.array(predicted_values)

ratios = predicted_outputs / actual_labels

# Calculate the mean and standard deviation
mean = np.mean(ratios)
std = np.std(ratios)

# Generate the range of x-values for the curve
x_values = np.linspace(mean - 6 * std, mean + 6 * std, 100)

# Compute the PDF using the mean and standard deviation
y_values = norm.pdf(x_values, mean, std)

# Create the figure and twin axes
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

# Create histogram
ax1.hist(ratios, bins=30, color='blue', alpha=0.7)
ax1.set_xlabel('Ratio (Predicted / Actual)')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram and Gaussian Distribution of Predicted / Actual Ratios')

# Plot the Gaussian distribution curve
ax2.plot(x_values, y_values, color='red', label='Gaussian Distribution')
ax2.set_ylabel('Probability Density')

# Show legend for the Gaussian distribution curve
ax2.legend(loc='upper right')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
plt.savefig('histogram_plot.png')
