import numpy as np
import matplotlib.pyplot as plt
import torch




# ------------------------------------

# for Bird classification
# maybe also for other image sets

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def plot_images(images, labels, classes, normalize = True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')

# 

# ------------------------------------







# ------------------------------------

# for svm
def plot_dataset(x, y, size=5):
    fig, ax = plt.subplots()
    ax.scatter(x[np.where(y==-1),0], x[np.where(y==-1),1], label='Class 1', s=size)
    ax.scatter(x[np.where(y==1),0], x[np.where(y==1),1], label='Class 2', s=size)
    ax.set_title('Training data')
    ax.legend()

# add data points
def plot_model_contour(model, x, y, x_min, x_max, size=5):
    grid_x, grid_y = torch.meshgrid(torch.arange(x_min*1.1, x_max*1.1, step=0.1),
                                torch.arange(x_min*1.1, x_max*1.1, step=0.1))
    x_test = torch.stack((grid_x, grid_y)).reshape(2, -1).transpose(1,0)

    y_test = model(x_test).detach()
    y_test = y_test.transpose(1,0).reshape(grid_x.shape).numpy()

    #fig, ax = plt.subplots(1,2, figsize=(8,3))
    fig, ax = plt.subplots()

    cs0 = ax.contourf(grid_x.numpy(), grid_y.numpy(), y_test)
    ax.contour(cs0, '--', levels=[0], colors='tab:green', linewidths=2)
    ax.plot(np.nan, label='decision boundary', color='tab:green')
    ax.scatter(x[np.where(y==-1),0], x[np.where(y==-1),1], s=size)
    ax.scatter(x[np.where(y==1),0], x[np.where(y==1),1], s=size)
    ax.legend()
    ax.set_title('Linear Kernel')
    
    ax.contour(cs0, '--', levels=[-1], colors='tab:gray', linestyles='dashed', linewidths=2)
    ax.contour(cs0, '--', levels=[1], colors='tab:gray', linestyles='dashed', linewidths=2)

    '''
    cs1 = ax[1].contourf(grid_x.numpy(), grid_y.numpy(), y_test_kernel)
    cs11 = ax[1].contour(cs1, '--', levels=[0], colors='tab:green', linewidths=2)
    ax[1].plot(np.nan, label='decision boundary', color='tab:green')
    ax[1].scatter(x[np.where(y==-1),0], x[np.where(y==-1),1])
    ax[1].scatter(x[np.where(y==1),0], x[np.where(y==1),1])
    ax[1].set_title('RBF Kernel')
    
    fig.subplots_adjust(wspace=0.2, hspace=0.1,right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.13, 0.02, 0.67])
    cbar = fig.colorbar(cs1, cax=cbar_ax)
    cbar.add_lines(cs11)
    
    ## try add margin
    cs12 = ax[1].contour(cs1, '--', levels=[-1], colors='tab:gray', linestyles='dashed', linewidths=2)
    cs13 = ax[1].contour(cs1, '--', levels=[1], colors='tab:gray', linestyles='dashed', linewidths=2)
    '''


# end svm

# ------------------------------------




# ------------------------------------

# for tensorflow playground

def w2plane(weight, bias):
    wx = weight[0]
    wy = weight[1]
    bias = bias[0]
    a = 0
    b = 0
    if_vertical = False
    if abs(wy) < 1e-8:
        if_vertical = True
        a = 1
        b = -bias/wx
    else:
        a = -wx/wy
        b = -bias/wy
    return a, b, if_vertical

def plot_play_data(data_0, data_1, scale=100, predict=False, hy_plane=None, if_vertical=False, max_line=1.1, min_line=-1.1):
    # need deal with vertical plane
    marker_list = ['o','*','^']
    color_list =  ['tab:blue', 'tab:orange', 'tab:green']
    fig, ax = plt.subplots()

    interior_c = 'none'

    x = data_0[:, 0]
    y = data_0[:, 1]
    i = 0
    if predict:
        interior_c = color_list[i]
    ax.scatter(x, y, c=interior_c, s=scale, label=color_list[i],
               alpha=0.3, edgecolors=color_list[i],
              marker=marker_list[i])
    
    
    x = data_1[:, 0]
    y = data_1[:, 1]
    i = 1
    if predict:
        interior_c = color_list[i]
    ax.scatter(x, y, c=interior_c, s=scale, label=color_list[i],
               alpha=0.3, edgecolors=color_list[i],
              marker=marker_list[i])

    if hy_plane != None:
        a, b = hy_plane[0], hy_plane[1]

        if if_vertical:
            ax.vlines(x=b, color='black', ymin=min_line, ymax=max_line)
        else:
            plot_x = np.linspace(min_line, max_line, 100)
            plot_y = a*plot_x + b

            ax.plot(plot_x, plot_y, color='black')
    
    ax.legend()
    ax.grid(True)
    
    plt.show()

# end tensorflow playground

# ------------------------------------