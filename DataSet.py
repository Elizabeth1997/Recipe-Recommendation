import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class RecipeRecommendation(Dataset):
	def __init__(self, csv_file_name, scale=False):
		self.file_name = csv_file_name
		self.scale = scale
		self.user_id, self.recipe_id, self.rating = self.get_data()
	
	def get_data(self):
		data = pd.read_csv(self.file_name)
		users = data['u'].to_numpy().reshape(-1,1)
		recipes = data['i'].to_numpy().reshape(-1,1)
		ratings = data['rating'].to_numpy().reshape(-1,1)
		if self.scale:
			ratings, _ = RecipeRecommendation.scale_min_max(ratings)
		#if self.is_train:
			# training process, indices to one-hot vectors
		#	onehot = OneHotEncoder(categories='auto',sparse=False)
		#	ratings = onehot.fit_transform(ratings)
		return users, recipes, ratings

	def __len__(self):
		size,_ = self.user_id.shape
		return size

	def __getitem__(self,idx):
		return self.user_id[idx], self.recipe_id[idx], self.rating[idx]

	@staticmethod
	def collate(batches):
		users_batch = torch.LongTensor([batch[0] for batch in batches])
		recipes_batch = torch.LongTensor([batch[1] for batch in batches])
		rating_batch = torch.Tensor([batch[2] for batch in batches])
		return users_batch, recipes_batch, rating_batch

	@staticmethod
	def scale_min_max(input_):
		scaler = MinMaxScaler()
		input_scaled = scaler.fit_transform(input_)
		return input_scaled, scaler
