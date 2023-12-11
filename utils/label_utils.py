import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyraws.utils.visualization_utils import equalize_tensor
import numpy as np



class image_analysis:
    # class that can analyse images
    # attributes -> compatible with equalize filter
    # methods    -> plot comparison (filtered - unfiltered image )
    #            -> calculate statistics ( rgb spectrum, avg rgb values, distance in color)

    def __init__(self, image_url):
        """Initialization.

        Args:
            image_url (string): path to the image

        """

        self.X = read_image(
            image_url
        ).type(
            torch.float32
        )  # read_image: PNG -> torch.tensor[image_channels, image_height, image_width]
        self.image_url = image_url
        self.Xequalized = torch.zeros((3, self.X.size(dim=1), self.X.size(dim=2))).type(
            torch.float32
        )
        self.boolTensor = True

    def plotTensor(self,dummyTensor):
        # TODO: adjust figure size
        # TODO: remove axis

        dummyTensor = dummyTensor.permute(1, 2, 0)
        # plotting

        plt.imshow(dummyTensor / dummyTensor.max())
        plt.axis("off")
        plt.show()

        # return dummyTensor.permute(2, 0, 1)
        return None
    

    def plotImage_url(self):


        dummyImage = mpimg.imread(self.image_url)
        plt.imshow(dummyImage)
        plt.axis("off")
        plt.show()

        return None

    def plotTensorEqualized(self):
        dummyTensor = self.X.permute(1, 2, 0)
        dummyTensor = dummyTensor.type(
            torch.float32
        )  # equalize_tensor requires floating point
        dummyTensor = equalize_tensor(dummyTensor)

        # self.Xequalized = dummyTensor

        # plt.imshow(dummyTensor/dummyTensor.max())
        # plt.axis('off')
        # plt.show()

        return dummyTensor.permute(2, 0, 1)

    def svd(self, r = 50):
        # use tensor
        # specify the reduction
                
        if self.boolTensor:
            dummyTensor = self.X

        else :
            dummyTensor = self.Xequalized
    
        tensor = torch.zeros((3,r,r))

        for i in range(3):
            u,s,vt = torch.linalg.svd(dummyTensor[i,:,:])
            # dummyTensor[i,:,:] = u[:r,:r]*torch.diag(s[:r])*torch.t(v[:r,:r])
            # print(dummyTensor[i,:,:])
            tensor[i,:,:] = torch.mm(torch.mm(u[:r,:r], torch.diag(s[:r])), vt[:r,:r])
            print(tensor)
        
        return tensor

    def RGB_distribution(self, dummyTensor):
        if self.boolTensor:
            dummyTensor = self.X

        else :
            dummyTensor = self.Xequalized

        reducedTensor = dummyTensor[:, :10000, :10000]
        reducedNumpy = torch.reshape(reducedTensor, (3, -1)).numpy()
        
        # calculating the histogram 
        hist_lst = []
        for i in range(3):

            # calculate histogram
            hist,bin_edges = np.histogram(reducedNumpy[i,:],bins=500)
           
            # get rid off rid of zero probabilities to avoid peaks in plot
            indZero =  np.where(hist==0)[0]
            bin_avg = np.zeros((hist.shape[0]- len(indZero)))
            hist_prob = np.zeros((hist.shape[0]- len(indZero)))

            # store histogram probalities and bin averages
            
            ind = 0 # other index to look filtered data set (without zero probabilities)
            for j in range(hist.shape[0]):
                
                if np.any(indZero == j): # if j in indZero, don't append
                    continue

                else :
                    bin_avg[ind] = (bin_edges[j] + bin_edges[j+1])/2  # average bin
                    hist_prob[ind] = np.round(hist[j] / reducedNumpy.shape[1],3) # probability of bin
                    ind = ind + 1

            hist_lst.append((bin_avg,hist_prob))


        # plotting the histogram -> 3 line plots for R,G,B
        colours = ["red","green","blue"]
        fig,ax = plt.subplots()
        for k in range(len(hist_lst)):
            ax.plot(hist_lst[k][0],hist_lst[k][1],color=colours[k])
        
        plt.show()
        return None




