from curses import flash
import sys
import os
import matplotlib as plt
from Network.Network import HomographyModel, HomographyModelUnsupervised, normalize
from Dataset.dataCreation import HomographyDataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np


# Don't generate pyc codes
sys.dont_write_bytecode = True
    
def TrainOperation(DirNamesTrain, DirNamesVal, NumEpochs, MiniBatchSize, CheckPointPath):
    """
    Inputs: 
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """ 

    
    trainDataset = HomographyDataset(DirNamesTrain, generate=False, transform=transform, name="train")
    valDataset = HomographyDataset(DirNamesVal, generate=False, transform=transform, name="val")

    trainDataloader = DataLoader(trainDataset, batch_size=MiniBatchSize)
    valDataloader = DataLoader(valDataset, batch_size=MiniBatchSize)

    model = HomographyModel()
    lossFunc = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum = 0.9)
    torch.autograd.set_detect_anomaly(True)
    # torch.autograd.set_detect_anomaly(True)

    loss_plot = np.zeros(NumEpochs)
    acc_plot = np.zeros(NumEpochs)
    epochs_plot = np.zeros(NumEpochs)

    for Epochs in range(NumEpochs):
        model.train()
        epochLoss = 0.
        unsupLoss = 0.
        counter = 0
        for iteration, (input, H_gt, ptsA, _) in enumerate(trainDataloader):
            counter = counter + 1

            H_pred = model(input)
            H_gt = H_gt.view(H_gt.shape[0],-1)
            loss = lossFunc(H_pred, H_gt)

            
            epochLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 50 == 0:
                epochLoss_avg = epochLoss / (iteration+1)
                with torch.no_grad():
                    model.eval()
                    valLoss = 0.
                    valLoss2 = 0.
                    for iteration_, (input, H_gt, ptsA, _) in enumerate(valDataloader):
                        input = input
                        H_gt = H_gt

                        H_pred = model(input)
                        H_gt = H_gt.view(H_gt.shape[0],-1)
                        loss = lossFunc(H_pred, H_gt)
                        valLoss += loss.item()
                    
                    valLoss_avg = valLoss / (iteration +1)
                    valLoss_avg2 = valLoss2 / (iteration_ +1)

                    print(f"Epoch : {Epochs}, Iter : {iteration}, Train Loss : {epochLoss_avg}, Val Loss : {valLoss_avg}, Val Loss 2 : {valLoss_avg2}")
                    
                model.train()

        loss_plot[Epochs] = epochLoss/counter
        epochs_plot[Epochs] = Epochs   
        print('Loss plot:=', loss_plot)
        print('Epoch plot:= ', epochs_plot)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(Epochs+1, NumEpochs, epochLoss/counter))
     
        torch.save(model.state_dict(), os.path.join(CheckPointPath, f"ckpt{Epochs}.pt"))
        
    plt.subplot(1,1,1)
    plt.xlim(0,NumEpochs)
    plt.ylim(0, 0.5)
    plt.title('Training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(epochs_plot, loss_plot)
    plt.savefig('Loss Plot sup.png')


def main():
    NumEpochs = 10
    trainPath = "/home/mihir/WPI/Fall 22/CV/project1/YourDirectoryID_p1/Phase2/Data/Train"
    valPath = "/home/mihir/WPI/Fall 22/CV/project1/YourDirectoryID_p1/Phase2/Data/Val"
    MiniBatchSize = 16
    CheckPointPath = os.path.join('../CheckPoints/', "Run1")
    
    TrainOperation(trainPath, valPath, NumEpochs, MiniBatchSize, CheckPointPath)
        
    
if __name__ == '__main__':
    main()

