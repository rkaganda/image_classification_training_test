import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable

data_dir = 'root'  # directory of images

# DATA SET LOADING


def load_split_train_test(_data_dir, valid_size=.2):
    # create transforms that will be used to "resize" the images
    _train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
    _test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])

    # apply the transforms to the training and test data
    train_data = datasets.ImageFolder(_data_dir, transform=_train_transforms)
    test_data = datasets.ImageFolder(_data_dir, transform=_test_transforms)

    # define the length of the training data and shuffle it
    num_train = len(train_data)  # the amount of training data
    indices = list(range(num_train))  # create a list of indices
    split = int(np.floor(valid_size * num_train))  # split the data
    np.random.shuffle(indices)  # shuffle the data

    # split the data set into training and testing by taking random samples of the data set
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    _trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    _testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    return _trainloader, _testloader


# generate the training and test data
trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)  # what classes are we training on

# use the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)  # define our model, pretrained for "faster" training
# print(model)

for param in model.parameters():
    param.requires_grad = False

# defining the layers of our model
model.fc = nn.Sequential(nn.Linear(2048, 512),  # linear transform of the data, 2048 input features, 512 output features
                         nn.ReLU(),  # rectified linear input layer for activation
                         nn.Dropout(0.2),  # .2 probability of zero'ing out elements of the input tensors
                         nn.Linear(512, 10),  # linear transform of the data, 512 input features, 10 output features
                         nn.LogSoftmax(dim=1))  # apply the logsoftmax gradient
criterion = nn.NLLLoss()  # negative log likely hood (smaller value = less likely)
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)  # optimize that shit boi
model.to(device)

epochs = 20  # how many times we train
steps = 0  # step counter
running_loss = 0  # the loss while we train
print_every = 10  # how many steps we output loss
train_losses, test_losses = [], []  # list to store the loss for cool plots

for epoch in range(epochs):  # for each training/testing iteration (epoch)
    for inputs, labels in trainloader:  # for each train set
        steps += 1  # increment the steps
        inputs, labels = inputs.to(device), labels.to(device)  # feed the inputs and labels to our model
        optimizer.zero_grad()  # zero out the gradient
        logps = model.forward(inputs)  # inputs forward
        loss = criterion(logps, labels)  # calculate our loss
        loss.backward()  # backpropagate the loss to LEARN
        optimizer.step()  # step the optimizer forward
        running_loss += loss.item()  # keep track of our loss

        if steps % print_every == 0:  # check if we display progress (loss)
            test_loss = 0  # init
            accuracy = 0  # init
            model.eval()  # we are evaluating the model NOT training
            with torch.no_grad():  # turn on gradation since we aren't training
                for _inputs, _labels in testloader:  # for each input, label in the test data set
                    _inputs, _labels = _inputs.to(device), _labels.to(device)  # feed the test data to our model
                logps = model.forward(_inputs)  # push the test "forward" through the model
                batch_loss = criterion(logps, _labels)  # check "loss" (how correct/ not correct) the model is
                test_loss += batch_loss.item()  # sum up the loss

                # who cares about accuracy, we just care about loss
                # ps = torch.exp(logps)
                # top_p, top_class = ps.topk(1, dim=1)
                # equals = top_class == labels.view(*top_class.shape)
            # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss / len(trainloader))  # append the running loss
            test_losses.append(test_loss / len(testloader))
            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. ")
                  # f"Test accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            model.train()  # train that biatch

torch.save(model, 'number_images.pth')  # done training the model, save it to a file

# show the cool graphs
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

# TESTING

data_dir = 'root'  # the directory of images
test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])  # create a resize tensor

# use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('number_images.pth')  # load the model
model.eval()


def predict_image(__image):
    image_tensor = test_transforms(__image).float()  # transform the image
    image_tensor = image_tensor.unsqueeze_(0)  # reshape the tensor
    __input = Variable(image_tensor)  # create to store image
    __input = __input.to(device)  # feed our input
    output = model(__input)  # get the result
    __index = output.data.cpu().numpy().argmax()  # fancy math to get the probability
    return __index


def get_random_images(num):
    # not going to comment this function
    # all it does shuffle the data set around then return random images as input/label
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    _images, __labels = dataiter.next()
    return _images, __labels


to_pil = transforms.ToPILImage()  # convert the transform to a PIL image
images, labels = get_random_images(5)  # get 5 random images from the data set

fig = plt.figure(figsize=(10, 10))  # create a plot to display our images

for ii in range(len(images)):  # for each image
    image = to_pil(images[ii])  # convert the image
    index = predict_image(image)  # predict what the image is
    sub = fig.add_subplot(1, len(images), ii+1)  # add a subplot that shows the image
    res = int(labels[ii]) == index  # display the label
    sub.set_title(str(trainloader.dataset.classes[index]) + ":" + str(res))  # display the "test" result in the title
    plt.axis('off')  # no axis for the plot
    plt.imshow(image)  # display the images in the plot


plt.show()  # show the plot


