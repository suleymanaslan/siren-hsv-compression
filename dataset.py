import torch
import imageio
import numpy as np

def get_mgrid(sidelen, dim):
    if dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        for i in range(dim):
            pixel_coords[..., i] = pixel_coords[..., i] / max(sidelen[i] - 1, 1)
        pixel_coords = torch.from_numpy((pixel_coords - 0.5) * 2).view(-1, dim)
        return pixel_coords.view((sidelen[0], sidelen[1], sidelen[2], dim))
    elif dim == 5:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2], :sidelen[3], :sidelen[4]], axis=-1)[None, ...].astype(np.float32)
        for i in range(dim):
            pixel_coords[..., i] = pixel_coords[..., i] / max(sidelen[i] - 1, 1)
        pixel_coords = torch.from_numpy((pixel_coords - 0.5) * 2).view(-1, dim)
        return pixel_coords.view((sidelen[0], sidelen[1], sidelen[2], sidelen[3], sidelen[4], dim))
    else:
        raise NotImplementedError(f'Not implemented for dim={dim}')

# class HyperspectralVideoData:
#     def __init__(self, data_folder):
#         self.data_folder = data_folder
        
#     def load_data(self, time, height, width, channels):
#         data = torch.ones((time, height, width, channels))
#         for ti in range(time):
#             for ci in range(channels):
#                 im = imageio.imread(f'{self.data_folder}/f{ti+1}/MSV_1_{ci+1}.png')
#                 im = torch.from_numpy(((im.astype(np.float32) / 255.0) - 0.5) / 0.5)
#                 data[ti,...,ci] = im
#         self.data = data
#         self.time = time
#         self.channels = channels
        
#         sidelength = (self.time, height, width)
#         mgrid = get_mgrid(sidelength, dim=3)
#         self.mgrid = mgrid
#         self.pixels = self.mgrid.view(-1, 3).shape[0]
#         assert self.pixels == self.data.view(-1, self.channels).shape[0]
        
#         print(f"Coords:{self.mgrid.shape}, Type:{self.mgrid.dtype}, Range:{(self.mgrid.min(), self.mgrid.max())}")
#         print(f"Data:{self.data.shape}, Type:{self.data.dtype}, Range:{(self.data.min(), self.data.max())}")
        
#     def get_pixels(self, batch_size):
#         ix = np.random.choice(self.pixels, batch_size)
#         batch_coord = self.mgrid.view(-1, 3)[ix]
#         batch_data = self.data.view(-1, self.channels)[ix]
#         return batch_coord, batch_data

#     def get_images(self, batch_size):
#         ix = np.random.choice(self.time, batch_size)
#         batch_coord = self.mgrid[ix,...]
#         batch_data = self.data[ix,...]
#         return batch_coord, batch_data

class MultispectralLightField:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        
    def load_data(self, scene):
        self.data = torch.from_numpy(((np.load(f'{self.data_folder}/{scene}.npy').astype(np.float32) / (2**16 - 1)) - 0.5) / 0.5)
    
        sidelength = (self.data.shape[0], self.data.shape[1], self.data.shape[2], self.data.shape[3], self.data.shape[4])
        self.mgrid = get_mgrid(sidelength, dim=5)
        self.pixels = self.mgrid.view(-1, 5).shape[0]
        assert self.pixels == self.data.view(-1, 1).shape[0]
        
        print(f"Coords:{self.mgrid.shape}, Type:{self.mgrid.dtype}, Range:{(self.mgrid.min(), self.mgrid.max())}")
        print(f"Data:{self.data.shape}, Type:{self.data.dtype}, Range:{(self.data.min(), self.data.max())}")
    
    def get_pixels(self, batch_size):
        ix = np.random.choice(self.pixels, batch_size)
        batch_coord = self.mgrid.view(-1, 5)[ix]
        batch_data = self.data.view(-1, 1)[ix]
        return batch_coord, batch_data
    
    def get_images(self, batch_size):
        yi = np.random.choice(9, batch_size)
        xi = np.random.choice(9, batch_size)
        li = np.random.choice(13, batch_size)
        batch_coord = self.mgrid[yi,xi,...,li,0:5]
        batch_data = self.data[yi,xi,...,li]
        return batch_coord, batch_data

