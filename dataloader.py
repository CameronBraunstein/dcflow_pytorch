
import os
import random
import numpy as np
from PIL import Image


def load_flow(flow_path,downsample_factor=1):
    with open(flow_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    data2D = data2D[::downsample_factor,::downsample_factor,:]
    data2D = data2D / downsample_factor
    return data2D

def load_image(image_path, downsample_factor=1):
    image = Image.open(image_path)
    height,width= image.size
    resize_shape = (height//downsample_factor,width//downsample_factor)
    image = image.resize(resize_shape, Image.Resampling.LANCZOS)
    return np.asarray(image)


def is_patch_center_in_bounds(center_y,center_x,height,width,padding):
    if ((center_y - padding < 0) or (center_y + padding + 1 > height)):
        return False
    if ((center_x - padding < 0) or (center_x + padding + 1 > width)):
        return False
    return True



def get_dataloader(args):
    if args.dataset == 'Sintel':
        dl = SintelDataloader(args)
    else:
        raise Exception('Only Sintel currently supported')
    return dl

class Dataloader:
    def __init__(self,args):
        self.mini_batch_size = args.mini_batch_size
        self.downsample_factor = args.downsample_factor
        self.patch_size = args.patch_size
        self.max_negative_offset = args.max_negative_offset
        self.min_negative_offset = args.min_negative_offset

        mini_batch_shape = (self.mini_batch_size,3,self.patch_size,self.patch_size)
        self.base_patches = np.zeros(mini_batch_shape).astype(np.float32)
        self.match_patches = np.zeros(mini_batch_shape).astype(np.float32)
        self.negative_patches = np.zeros(mini_batch_shape).astype(np.float32)

    def get_patch_mini_batch(self):
        return self.base_patches, self.match_patches, self.negative_patches
    
class SintelDataloader(Dataloader):
    def __init__(self,args):
        super(SintelDataloader, self).__init__(args)
        sintel_root_path = os.path.join('datasets','Sintel', args.training_or_test)
        self.flow_root_path = os.path.join(sintel_root_path,'flow')
        self.scenes_root_path = os.path.join(sintel_root_path, args.clean_or_final)
        if args.action == 'Train':
            # "We randomly select 14 sequences from the fi-
            # nal rendering pass for training, and use the remaining 9 se-
            # quences as a validation set"
            number_training_scenes = 14
            scene_names = os.listdir(self.scenes_root_path)
            random.shuffle(scene_names)
            #self.training_scene_paths = [os.path.join(training_base_path, scene_name) for scene_name in scene_names[:number_training_scenes]]
            #self.validation_scene_paths = [os.path.join(training_base_path, scene_name) for scene_name in scene_names[number_training_scenes:]]

            self.training_scene_names = scene_names[:number_training_scenes]
            self.validation_scene_names = scene_names[number_training_scenes:]
        elif args.action == 'CostVolumeAccuracy':
            self.frame_0_path, self.frame_1_path,self.flow_path = self.format_frame_pair_and_flow_paths(args.scene_name,args.first_frame_number)

    def format_frame_pair_and_flow_paths(self,scene_name,first_frame_number):
        frame_0_path = os.path.join(self.scenes_root_path,scene_name,'frame_{0:04d}.png'.format(first_frame_number))
        frame_1_path = os.path.join(self.scenes_root_path,scene_name,'frame_{0:04d}.png'.format(first_frame_number+1))
        flow_path = os.path.join(self.flow_root_path,scene_name,'frame_{0:04d}.flo'.format(first_frame_number))
        return frame_0_path,frame_1_path,flow_path
    
    def get_frame_pair(self):
        frame_0 = load_image(self.frame_0_path, self.downsample_factor)
        frame_1 = load_image(self.frame_1_path, self.downsample_factor)
        return frame_0,frame_1

    def prepare_patch_mini_batch(self,is_validation=False):
        remaining_triples = self.mini_batch_size-1
        while(remaining_triples>=0):
            if is_validation:
                scene_name = random.choice(self.validation_scene_names)
            else:
                scene_name = random.choice(self.training_scene_names)
            training_scene_path = os.path.join(self.scenes_root_path,scene_name)

            base_scene_number = random.randint(1, len(os.listdir(training_scene_path))-1)
            base_file,match_file,flow_file = self.format_frame_pair_and_flow_paths(scene_name,base_scene_number)
            base_image = load_image(base_file,self.downsample_factor)
            flow = load_flow(flow_file,self.downsample_factor)
            height, width, _ = base_image.shape
            padding = self.patch_size // 2
            h_1 = np.random.randint(padding, height - padding)
            w_1 = np.random.randint(padding, width - padding)
            h_2 = int(np.round(h_1 + flow[h_1,w_1,0]))
            w_2 = int(np.round(w_1 + flow[h_1,w_1,1]))
            if is_patch_center_in_bounds(h_2,w_2,height,width,padding):
                base_patch = base_image[h_1 - padding: h_1 + padding + 1 ,w_1 - padding: w_1 + padding + 1,:]
                match_image = load_image(match_file,self.downsample_factor)
                match_patch = match_image[h_2 - padding: h_2 + padding+1,w_2 - padding: w_2 + padding+1,:]
                negative_found = False
                while(not negative_found):
                    height_offset = h_2 + np.random.randint(self.min_negative_offset,self.max_negative_offset)*(1-2*np.random.randint(2))
                    width_offset = w_2 + np.random.randint(self.min_negative_offset,self.max_negative_offset)*(1-2*np.random.randint(2))
                    if is_patch_center_in_bounds(height_offset,width_offset,height,width,padding):
                        negative_patch = match_image[height_offset- padding: height_offset + padding + 1,width_offset - padding: width_offset + padding + 1,:]
                        negative_found = True

                self.base_patches[remaining_triples,:,:,:] = np.transpose(base_patch,(2,0,1))
                self.match_patches[remaining_triples,:,:,:] = np.transpose(match_patch,(2,0,1))
                self.negative_patches[remaining_triples,:,:,:] = np.transpose(negative_patch,(2,0,1))

                remaining_triples = remaining_triples - 1
    
    def enqueue_batch_samples(self,multiprocess_queue,is_validation=False):
        remaining_triples = self.mini_batch_size-1
        mini_batch_shape = (self.mini_batch_size,3,self.patch_size,self.patch_size)
        base_patches = np.zeros(mini_batch_shape).astype(np.float32)
        match_patches = np.zeros(mini_batch_shape).astype(np.float32)
        negative_patches = np.zeros(mini_batch_shape).astype(np.float32)

        while(remaining_triples>=0):
            if is_validation:
                scene_name = random.choice(self.validation_scene_names)
            else:
                scene_name = random.choice(self.training_scene_names)
            training_scene_path = os.path.join(self.scenes_root_path,scene_name)

            base_scene_number = random.randint(1, len(os.listdir(training_scene_path))-1)
            base_file,match_file,flow_file = self.format_frame_pair_and_flow_paths(scene_name,base_scene_number)
            base_image = load_image(base_file,self.downsample_factor)
            flow = load_flow(flow_file,self.downsample_factor)
            height, width, _ = base_image.shape
            padding = self.patch_size // 2
            h_1 = np.random.randint(padding, height - padding)
            w_1 = np.random.randint(padding, width - padding)
            h_2 = int(np.round(h_1 + flow[h_1,w_1,0]))
            w_2 = int(np.round(w_1 + flow[h_1,w_1,1]))
            if is_patch_center_in_bounds(h_2,w_2,height,width,padding):
                base_patch = base_image[h_1 - padding: h_1 + padding + 1 ,w_1 - padding: w_1 + padding + 1,:]
                match_image = load_image(match_file,self.downsample_factor)
                match_patch = match_image[h_2 - padding: h_2 + padding+1,w_2 - padding: w_2 + padding+1,:]
                negative_found = False
                while(not negative_found):
                    height_offset = h_2 + np.random.randint(self.min_negative_offset,self.max_negative_offset)*(1-2*np.random.randint(2))
                    width_offset = w_2 + np.random.randint(self.min_negative_offset,self.max_negative_offset)*(1-2*np.random.randint(2))
                    if is_patch_center_in_bounds(height_offset,width_offset,height,width,padding):
                        negative_patch = match_image[height_offset- padding: height_offset + padding + 1,width_offset - padding: width_offset + padding + 1,:]
                        negative_found = True

                base_patches[remaining_triples,:,:,:] = np.transpose(base_patch,(2,0,1))
                match_patches[remaining_triples,:,:,:] = np.transpose(match_patch,(2,0,1))
                negative_patches[remaining_triples,:,:,:] = np.transpose(negative_patch,(2,0,1))

                remaining_triples = remaining_triples - 1
        multiprocess_queue.put((base_patches,match_patches,negative_patches))