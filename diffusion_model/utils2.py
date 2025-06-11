import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import math

"""Uncomment to download dataset"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_CIFAR10(num_classes=10):
    
    transform = transforms.Compose(
        [transforms.ToTensor()]) 
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if num_classes < 10:
        # Filter trainset and testset to include only the specified number of classes
        train_indices = [i for i, (_, label) in enumerate(trainset) if label < num_classes]
        test_indices = [i for i, (_, label) in enumerate(testset) if label < num_classes]

        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)

        # Update classes list
        classes = classes[:num_classes]

    return trainset, testset, classes
def getSubset(dataset, nr_img):
    dog_indices, deer_indices, car_indices, plane_indices, frog_indices = [], [], [], [], []
    truck_indices, cat_indices, bird_indices, horse_indices, ship_indices = [], [], [], [], []
    dog_idx, deer_idx = 0,1
    car_idx, plane_idx = 2,3
    frog_idx, truck_idx = 4,5
    cat_idx, bird_idx = 6,7
    horse_idx, ship_idx = 8,9

    for i in range(len(dataset)):
        current_class = dataset[i][1]
        #print(current_class)
        if current_class == dog_idx:
            dog_indices.append(i)
        elif current_class == deer_idx:
            deer_indices.append(i)
        elif current_class == car_idx:
            car_indices.append(i)
        elif current_class == plane_idx:
            plane_indices.append(i)
        elif current_class == frog_idx:
            frog_indices.append(i)
        elif current_class == truck_idx:
            truck_indices.append(i)
        elif current_class == cat_idx:
            cat_indices.append(i)
        elif current_class == bird_idx:
            bird_indices.append(i)
        elif current_class == horse_idx:
            horse_indices.append(i)
        elif current_class == ship_idx:
            ship_indices.append(i)
        dog_indices = dog_indices[:nr_img]
        deer_indices = deer_indices[:nr_img]
        plane_indices = plane_indices[:nr_img]
        frog_indices = frog_indices[:nr_img]
        truck_indices = truck_indices[:nr_img]
        cat_indices = cat_indices[:nr_img]
        bird_indices = bird_indices[:nr_img]
        horse_indices = horse_indices[:nr_img]
        ship_indices = ship_indices[:nr_img]
        car_indices = car_indices[:nr_img]
        new_dataset = Subset(dataset, dog_indices+plane_indices+deer_indices+frog_indices+car_indices+ship_indices+horse_indices+bird_indices+cat_indices+truck_indices)
    return new_dataset
def show_images(img):
    # img = img.cpu().detach().numpy()  # Move to CPU and convert to NumPy array
    img = img.detach().cpu().numpy()
    rows = math.floor(len(img)**0.5)
    cols = rows
    fig, ax = plt.subplots(rows,cols)
    
    for i,image in enumerate(img):
        row = math.floor(i/rows)
        col = i%cols

        im = ax[col,row].imshow(image.squeeze(0).permute(1,2,0))
    plt.show()


# def show_images(img):
#     batch_size, height, width, channels = img.shape  # Adjust the shape to match NumPy format
#     rows = math.ceil(batch_size / 4)
#     cols = 4
#     fig, ax = plt.subplots(rows, cols, figsize=(10, 10))
#     fig.subplots_adjust(hspace=0.1, wspace=0.1)
#     for i in range(rows * cols):
#         row = i // cols
#         col = i % cols
#         if i < batch_size:
#             image = img[i]
#             ax[row, col].imshow(image)
#         ax[row, col].axis('off')
#     plt.show()