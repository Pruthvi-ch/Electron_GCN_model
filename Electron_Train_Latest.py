import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader, Dataset, Data
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import time
# with open('/content/drive/MyDrive/Pickle_File/Electron.pkl', 'rb') as f:
#     data_list = pickle.load(f)
import glob


#file_paths = ['Electron_latest_p5mip_0.pkl', 'Electron_latest_p5mip_1000.pkl', 'Electron_latest_p5mip_2000.pkl', 'Electron_latest_p5mip_3000.pkl', 'Electron_latest_p5mip_4000.pkl', 'Electron_latest_p5mip_5000.pkl', 'Electron_latest_p5mip_6000.pkl', 'Electron_latest_p5mip_7000.pkl', 'Electron_latest_p5mip_8000.pkl', 'Electron_latest_p5mip_9000.pkl', 'Electron_latest_p5mip_10000.pkl', 'Electron_latest_new_0.pkl', 'Electron_latest_new_13500.pkl', 'Electron_latest_new_16500.pkl', 'Electron_latest_new_6000.pkl', 'Electron_latest_new_10500.pkl', 'Electron_latest_new_1500.pkl', 'Electron_latest_new_3000.pkl', 'Electron_latest_new_7500.pkl', 'Electron_latest_new_12000.pkl', 'Electron_latest_new_15000.pkl', 'Electron_latest_new_4500.pkl', 'Electron_latest_new_9000.pkl']
file_paths = glob.glob('Electron_latest_PU_000_60k_*.pkl')



data_list = []
 # Load the data from the pickle file
for file_path in file_paths:
    with open(file_path, "rb") as f:
        data_list += pickle.load(f)

print(len(data_list))
# Define lists to store evaluation results
all_preds = []
all_labels = []
all_losses = []
accuracies = []
torch.manual_seed(69)
#exit()
#train_data = data_list[0:80]
#test_data = data_list[80:100]

train_data = data_list[:23000]
test_data = data_list[23000:25000]

# 80:20 split for 'n' number of graphs
#num_test_graph=20000
#train_data = data_list[0:int(4*num_test_graph/5)]
#test_data = data_list[int(4*num_test_graph/5):num_test_graph]

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear
from torch_geometric.data import DataLoader

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

reg_lambda = 0.0001

# Define the training parameters
num_features = 4
hidden_channels = 16
lr = 0.00004
epochs = 1000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create the GCN model and optimizer
#model = Net(num_features, hidden_channels)
model=torch.load("Electron_PU_000_60k.pt")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = MultiStepLR(optimizer, milestones=[30, 50, 100, 150, 200, 300, 400, 500, 700], gamma=0.5)
model.to(device)

# Create data loaders for the training and testing datasets
batch_size = 1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

q = 0.5  # Choose the desired quantile value
criterion = torch.nn.SmoothL1Loss()

import sys

predictions = []
targets = []
min_loss_epoch=0;
min_mse_epoch=0;

min_loss=sys.maxsize
min_mse=sys.maxsize

loss_values = []  # List to store loss values
epochs_list = []

all_pred_data=[]
train_times = []  # To store time taken for each epoch during training
test_times = []  # To store time taken for each epoch during testing

# Train the model
for epoch in range(epochs):
    predictions = []
    targets = []
    tuples_epoch=[]
    train_start = time.time()
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        batch1 = batch.batch.to(device)
        y1 = batch.y[:4].to(device)
        y1 = torch.unsqueeze(y1, 0)
        #print(y1.size())
        out = model(x, edge_index, y1, batch1)
        out = out.squeeze()  # Remove the extra dimensions
        target = batch.y[4].float().to(device)  # Convert target values to float
        #print(target)
        loss = criterion(out, target)
        l1_reg = 0
        for param in model.parameters():
            l1_reg += torch.abs(param).sum()
        #loss = torch.mean(torch.where(torch.abs(out - target) < q, 0.5 * (out - target) ** 2, q * torch.abs(out - target) - 0.5))
        loss = torch.mean((10000*((out/target) - 1)) ** 2)
        loss.backward()
        optimizer.step()
    print('Epoch {}, Loss: {}'.format(epoch+1, loss.item()))
    train_end = time.time()
    epoch_train_time = train_end - train_start
    train_times.append(epoch_train_time)
    print("Training Time for Epoch: {:.3f} seconds".format(epoch_train_time))

    model.eval()
    
    test_start = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch1 = batch.batch.to(device)
            y1 = batch.y[:4].to(device)
            y1 = torch.unsqueeze(y1, 0)
            out = model(x, edge_index, y1, batch1)
            predictions.extend(out.cpu().numpy().flatten().tolist())
            targets.extend(batch.y[4].cpu().numpy().flatten().tolist())
            out = out.squeeze()
            target = batch.y[4].float().to(device)
            #loss = criterion(out, target)
            loss = torch.mean((10000*((out/target) - 1) ** 2))
            total_loss += loss.item()
            tuple_data=(out,target)
            tuples_epoch.append(tuple_data)
            #print('Actual label was->',batch.y,'Predicted output:', out)

    mean_loss = total_loss / len(test_loader.dataset)
    if(min_loss>mean_loss):
        min_loss=mean_loss
        min_loss_epoch=epoch+1
        torch.save(model,'Electron_PU_000_60k_2.pt')
    loss_values.append(mean_loss) # Append the loss value to the list
    epochs_list.append(epoch + 1)  # Append the epoch number to the list


    mse = mean_squared_error(targets, predictions)

    if(mse<min_mse):
      min_mse=mse
      min_mse_epoch=epoch+1
      #torch.save(model,'Electron_PU_000_60k.pt')


    print('Mean Loss: {}'.format(mean_loss))

    print('Mean Squared Error (MSE): {:.2f}'.format(mse))
    print('Root mean Squared Error (RMSE) : {:.2f}'.format(np.sqrt(mse)))
    all_pred_data.append(tuples_epoch)
    test_end = time.time()
    epoch_test_time = train_end - train_start
    test_times.append(epoch_test_time)
    #print("Testing Time for Epoch: {:.3f} seconds".format(epoch_test_time))


predictions = np.array(predictions)
targets = np.array(targets)


# # Scatter plot with different colors for predicted and actual values
'''
plt.scatter(targets, predictions, color='red', label='Predicted')
plt.scatter(targets, targets, color='blue', label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.legend()
plt.show()
'''
# Assuming you have a list of energy values
# Generate x-axis positions based on the index of each value in the list
x_positions = list(range(len(targets)))

# Plot the data
# plt.plot(x_positions, targets, marker='o', linestyle='-', color='blue')
# # plt.scatter(x_positions, predictions, color='red',linestyle='-', label='Predicted')
# # Set labels and title
# plt.xlabel('Position in the List')
# plt.ylabel('Energy Values')
# plt.title('Energy Values vs. Position')



# Loss vs epoch

# Display the plot
# plt.plot(epochs, loss_values, marker='o', linestyle='-')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss vs. Epoch')
# plt.show()


# Calculate evaluation metrics
#mse = mean_squared_error(targets, predictions)
min_rmse = np.sqrt(min_mse)
mae = mean_absolute_error(targets, predictions)
# r2 = r2_score(targets, predictions)

print('-------------------------------------')
print('-------------------------------------')
print('Min Mean Squared Error (MSE): {:.2f}'.format(min_mse))
print('Min Root Mean Squared Error (RMSE): {:.2f}'.format(min_rmse))
# print('Mean Absolute Error (MAE): {:.2f}'.format(mae))
# print('R-squared (R2): {:.2f}'.format(r2))


print("min loss epoch is ",min_loss_epoch)
print("min rmse epoch is ",min_mse_epoch)

avg_train_time = sum(train_times) / len(train_times)
avg_test_time = sum(test_times) / len(test_times)

print("Average Training Time per Epoch: {:.5f} seconds".format(avg_train_time))
print("Average Testing Time per Epoch: {:.5f} seconds".format(avg_test_time))

with open("predictions_values.pkl","wb") as f:
    pickle.dump(all_pred_data,f)

#torch.save(model,'/home/4tb_Drive_1/HGCAL_HLT/WeightElectronFile.pt')

for x in train_times:
    print(x)
print("-------------------------------------------------------------------")
for k in test_times:
    print(k)
