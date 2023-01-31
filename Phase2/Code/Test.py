import sys
from Network.Network import HomographyModel, HomographyModelUnsupervised, normalize
from Dataset.dataCreation import HomographyDataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import cv2
import numpy as np


# Don't generate pyc codes
sys.dont_write_bytecode = True

def TestOperation(ModelPath, ModelType, BasePath, MiniBatchSize):


	spath=f"../viz/{ModelType}/test"
	if ModelType == 'Sup':
		model = HomographyModel()
	elif ModelType == 'Unsup':
		model = HomographyModelUnsupervised()
	
	lossFunc = nn.MSELoss()

	model.load_state_dict(torch.load(ModelPath))

	for name in ["test"]:
		test_dataset = HomographyDataset(BasePath, generate=True, transform=transform, name=name)
		test_dataloader = DataLoader(test_dataset, batch_size=MiniBatchSize)

		with torch.no_grad():
			model.eval()
			valLoss = 0.
			for idx1, (input, H_gt, ptsA, IA) in enumerate(test_dataloader):
				input = input
				H_gt = H_gt
				if ModelType == 'Sup':
					H_pred = model(input)
					H_gt = H_gt.view(H_gt.shape[0],-1)
				if ModelType == 'Unsup':
					ptsA = ptsA
					pA, pB = torch.chunk(input, dim=1, chunks=2)
					_, H_pred = model(input, ptsA, pA)
				
				loss = lossFunc(H_pred, H_gt)
				valLoss += loss.item()

			
			for j in range(IA.shape[0]):
				img = IA[j]
				img = img.permute(1,2,0).numpy()
				img = img - img.min()
				img = img / img.max()
				img = (img*255).astype(np.uint8)
				base_pts= (ptsA[j]).numpy().reshape(-1,1,2).astype(np.int32)
				gt_pts= (H_gt[j]*32).numpy().reshape(-1,1,2).astype(np.int32)
				pred_pts= (H_pred[j]*32).numpy().reshape(-1,1,2).astype(np.int32)
				gt_pts = gt_pts + base_pts
				pred_pts = pred_pts + base_pts

				img = cv2.polylines(img.copy(), [gt_pts], True, (0,255,0), 2)
				img= cv2.polylines(img, [pred_pts], True, (255,0,0), 2)

				cv2.imwrite(f"{spath}/{j}.jpg", img)
					
			valLoss_avg = valLoss / (idx1 +1)
		
		print(f"{name} Dataset, Average Loss : {valLoss_avg}")
		

def main():

	ModelPath = '../CheckPoints/Unsup/Run1/ckpt9.pt'
	BasePath = '/home/mihir/WPI/Fall 22/CV/project1/YourDirectoryID_p1/Phase2/Data/Train/'
	ModelType = 'Sup'

	MiniBatchSize = 16

	TestOperation(ModelPath, ModelType, BasePath, MiniBatchSize)
	 
if __name__ == '__main__':
	main()
 
