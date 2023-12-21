#This replaces the code form the notebook provided from the initialization of the dataset to and including the dataLoader functions. It should be a drop in
#replacement as the variable names do not change and the final result should have the same form as before

from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np

batch_size=16 #sampling batch size, keep it as factors of to the power of 2

#Width and height of the images provided, this is consistent throughout the dataset and should not be changed
image_width = 1024 
image_height = 1942

#To what % of the original size the image is downscaled in the transformation. This is done to speed up the network and reduce resources needed
scale_factor = 0.4

#Dataset split sizes, this is taken from the notebook provided and should not be changed
valid_size = 0.15
test_size = 0.15

#The mean and standard deviation of all three color channels in the dataset. This is used to normalize the dataset, which is a basic transform 
#in all networks and should be kept in. Do not change these values unless the dataset changes. The values are currently based on the 60 image 
#sample i had so they might be updated to the entire dataset values in the future.
dataset_mean = [0.2391, 0.4028, 0.4096]
dataset_std = [0.2312, 0.3223, 0.3203]

#This variable includes all the transformations done on the dataset. If there are any anyone wants to add, add them below "Resize" as these transformation
#should be performed first.
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((dataset_mean),(dataset_std)),
                                transforms.Resize((int(image_width*scale_factor), int(image_height*scale_factor)))])

#Reading the dataset, here the function used is changed from the notebook provided.
dataset = datasets.ImageFolder(root=path_data, 
                                 transform=transform)

#Based on the number of images included, this determines the number of images in each split based on the distribution provided above.
n_val = int(np.floor(valid_size * len(dataset)))
n_test = int(np.floor(test_size * len(dataset)))
n_train = len(dataset) - n_val - n_test

#Splitting the dataset
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

# Train loader
train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=False, shuffle=True)
# Cross validation data loader
valid_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=False, shuffle=True)
# Test data loader
test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=False, shuffle=True)

#This is here just as a reference since named labels are lost with this approach (just how the function works, might be a workaround but not worth it)
#0 - cloud
#1 - edge
#2 - good


#Just printing the distribution of samples in each split.
unique, counts = np.unique(torch.tensor([train_ds.dataset.targets[i] for i in train_ds.indices]), return_counts=True)
print("Train split: ", dict(zip(unique, counts)))

unique, counts = np.unique(torch.tensor([test_ds.dataset.targets[i] for i in test_ds.indices]), return_counts=True)
print("Test split: ", dict(zip(unique, counts)))

unique, counts = np.unique(torch.tensor([val_ds.dataset.targets[i] for i in val_ds.indices]), return_counts=True)
print("Validation split: ", dict(zip(unique, counts)))
