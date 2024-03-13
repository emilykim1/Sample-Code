import torch
import numpy as np
import math
import random
import os
import PIL.Image
import imageio
import cv2
from tqdm import tqdm
import torchvision
import pickle


from torchvision.utils import save_image
import torchvision.transforms as transforms

from training.attack_losses import ScoreLoss, TVLoss, NPSLoss, DiversionLoss, DiscriminatorLoss

from detectron2.utils.events import EventStorage
from detectron2.structures import Boxes
from detectron2.structures.instances import Instances
from torch.utils.tensorboard import SummaryWriter

from render_images import differentiablerenderer

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
#from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from model_zoo.RetinaNetPoint import RetinaNetPoint

from training.attack_dataset import AttackDataset, collate_fn


# seed = 0
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)


def blur_objects(gaussian_blurrer, rendered_images, background_images):
    # Find the difference between the images
    diff_images = rendered_images - background_images
    
    # Blur the difference
    diff_images = gaussian_blurrer(diff_images)
    
    # Add the blurred difference images to the background images
    blurred_images = background_images + diff_images
    
    # Convert to float
    blurred_images = blurred_images.float()
    
    return blurred_images

def init_adv(adv_textures, requires_grad=True):
    """
    Initialize an adversarial texture map. If config file contains path to a previously saved map -- it is loaded.
    
    outputs:
        - adv_textures (torch.Tensor): a tensor of shape (H, W, 3) which represents a unified adversarial texture map
    """
        
    if requires_grad:
        adv_textures.requires_grad_(True)
    
    return adv_textures

def init_optimizer_scheduler(optimized_params):
    optimizer = torch.optim.SGD(optimized_params, lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    return optimizer, scheduler

def setup_loss_fns():
    loss_fns = {}
    loss_fns_list = ["native"]
    
    # Check that the loss terms are valid
    loss_fns_set = set(loss_fns_list)
    pure_set = set(("yolov5", "native"))
    attack_set = set(("scoreloss", "diversion"))
    loss2pure = len(loss_fns_set.intersection(pure_set))
    loss2attack = len(loss_fns_set.intersection(attack_set))
    valid_intersection = not ((loss2pure > 0) and (loss2attack > 0)) # Check that loss terms do not contain elements from both pure and attack sets
    assert valid_intersection, f"Loss terms combination is invalid! Cannot use loss terms from the pure set ({pure_set}) and the attack set ({attack_set}) at the same time!"
    
    # Construct the loss dictionary
    for loss_keyword in loss_fns_list:
        if loss_keyword == 'scoreloss':
            coefficient = 1.0
            loss = ScoreLoss(coefficient=coefficient)
        elif loss_keyword == 'TV':
            coefficient = 2000.0
            loss = TVLoss(coefficient=coefficient)
        # elif loss_keyword == 'NPS':
        #     coefficient = 4.0
        #     #printable_colors = #torch.load(self.attack_cfg.ATTACKER.OPTIMIZATION.LOSS.FILES.NPS_COLORS)
        #     loss = NPSLoss(printable_colors, coefficient=coefficient, device=dev)
        elif loss_keyword == 'diversion':
            coefficient = 1.0
            loss = DiversionLoss(coefficient=coefficient)
        elif loss_keyword == 'discriminator':
            coefficient = 1.0
            loss = DiscriminatorLoss(discriminator=self.discriminator, coefficient=coefficient)
        elif loss_keyword == 'native':
            loss = None
        else:
            raise NotImplementedError
        loss_fns[loss_keyword] = loss
    return loss_fns


def construct_model_inputs(synthetic_images_batch, batch_size, img_res, image_id):
    assert synthetic_images_batch.shape[0] == batch_size

    empty_gt = True
    
    model_inputs = []
    for i in range(batch_size):
        model_input = {}
        model_input['image_id'] = image_id
        model_input['height'] = img_res
        model_input['width'] = img_res
        model_input['image'] = synthetic_images_batch[i] * 255.

        # Construct the instances field
        if empty_gt:
            gt_classes = torch.empty(0)
            gt_boxes = torch.empty(size=(0, 4))
        else:
            gt_classes = torch.tensor([0] * len(labels_batch[i]))
            gt_boxes = torch.empty(size=(labels_batch[i].shape[0], 4))
            gt_boxes[:, :2] = labels_batch[i] - 24 // 2
            gt_boxes[:, 2:] = labels_batch[i] + 24 // 2
            gt_boxes.data.clamp_(0, img_res)
        gt_boxes = Boxes(gt_boxes)
        instances = Instances(
            (img_res, img_res),
            gt_boxes=gt_boxes,
            gt_classes=gt_classes
        )
        model_input['instances'] = instances

        # Append to the list
        model_inputs.append(model_input)

        # Modify image ID
        image_id += 1
    
    
    return model_inputs

def losses_forward(results, model_inputs, loss_fns, adv_textures):
    losses_dict = {}
    for loss_fn_keyword in loss_fns.keys():
        if loss_fn_keyword == 'scoreloss':
            loss = loss_fns[loss_fn_keyword](results)
        elif loss_fn_keyword == 'TV':
            loss = loss_fns[loss_fn_keyword](adv_textures.unsqueeze(0).permute(0, 3, 1, 2))
        elif loss_fn_keyword == 'NPS':
            loss = loss_fns[loss_fn_keyword](adv_textures.unsqueeze(0))
        elif loss_fn_keyword == 'diversion':
            gt_points_batch = [model_input['instances'].gt_boxes.get_centers().to(self.attack_cfg.DEVICE) for model_input in model_inputs]
            pred_points_batch = [result.pred_boxes.get_centers() for result in results]
            loss = loss_fns[loss_fn_keyword](pred_points_batch, gt_points_batch)
        elif loss_fn_keyword == 'discriminator':
            images_batch = torch.stack([model_input['image'] for model_input in model_inputs]) / 255.
            images_batch = None#self.normalizer(images_batch)
            loss = loss_fns[loss_fn_keyword](images_batch)
        elif loss_fn_keyword == 'native':
            loss = sum(results.values())
        losses_dict[loss_fn_keyword] = loss
    return losses_dict

def log_info(iter_counter, loss_dict, total_loss, results, writer, scheduler):
    for k, v in loss_dict.items():
        writer.add_scalar(k, v.item(), iter_counter)
        writer.add_scalar("Total loss", total_loss, iter_counter)
        #writer.add_scalar("LR", scheduler.get_last_lr()[0], iter_counter)
        #if self.forward_mode == 'attack':
        #writer.add_scalar("Maximum score", max([torch.max(result.scores) for result in results if result.scores.shape[0] > 0]).item(), iter_counter)

def log_params(iter_counter, sphere_vector_grad, writer):
    writer.add_scalar("sphere_vector gradient", sphere_vector_grad, iter_counter)
    # writer.add_scalar("Length zeroth dimension", l.item(), iter_counter)
    # writer.add_scalar("DX", dx.item(), iter_counter)
    # writer.add_scalar("DY", dy.item(), iter_counter)

def update_LR(scheduler, lr_decay_rate, epoch):
    if (epoch + 1) % lr_decay_rate == 0:
        scheduler.step()

def blur_objects(gaussian_blurrer, rendered_images, background_images):
    # Find the difference between the images
    diff_images = rendered_images - background_images
    
    # Blur the difference
    diff_images = gaussian_blurrer(diff_images)
    
    # Add the blurred difference images to the background images
    blurred_images = background_images + diff_images
    
    # Convert to float
    blurred_images = blurred_images.float()
    
    return blurred_images

def narrow_gaussian(x, ell):
    return torch.exp(-0.5 * (x / ell) ** 2)

def approx_count_zero(x, ell=1e-3):
    # Approximation of || x ||_0
    return narrow_gaussian(x, ell).sum(dim=-1)


def soft_argmax(core_textures, temp=0.06):
        """
        This function implements one-hot softmax approximation of argmax (using low temperature softmax).
        Exact formula:
        
        z_i = exp(ln(x_i) / temp) / [sum_j(exp(ln(x_j) / temp))]
        """
        numerator = torch.exp(torch.log(core_textures) / temp)
        denominator = torch.sum(numerator, dim=1)
        soft_argmax = numerator / denominator.reshape(numerator.shape[0],1)
        
        return soft_argmax
    


def attack_train_cube(detector_attack_model, attack_loader, attack_logdir, device=None):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    img_res = 384


    dif = differentiablerenderer(device=device)

    M = dif.load_meshes(meshes_dir = "/home/ekim2/Storage/satellite-project/meshes/GAN-vehicles/", device=device)
    
    # set number of spheres
    num_spheres = 1
    
#     scale = torch.nn.parameter.Parameter(init_adv(0.5*torch.ones(num_spheres,device=device)))
    
#     color = torch.nn.parameter.Parameter(init_adv(torch.rand([num_spheres,5],device=device)))

    init_vertex = 212
    mapping_vertex = np.load('/home/ekim2/Storage/satellite-project/meshes/004_281to962_vertex_indices_mapping.npy')[init_vertex]
    
    location = torch.nn.parameter.Parameter(init_adv(M[0].verts_packed()[mapping_vertex].clone().reshape(1,3)))
    
    delta = 0.1
    
    # min_x = torch.min(M[0].verts_packed()[:,0]) - delta
    # max_x = torch.max(M[0].verts_packed()[:,0]) + delta
    # min_z = torch.min(M[0].verts_packed()[:,2]) - delta
    # max_z = torch.max(M[0].verts_packed()[:,2]) + delta
    # min_y = torch.min(M[0].verts_packed()[:,1]) - delta
    # max_y = torch.max(M[0].verts_packed()[:,1]) + delta
    
    min_x = location[0,0] - delta
    max_x = location[0,0] + delta
    min_y = location[0,1] - delta
    max_y = location[0,1] + delta
    min_z = location[0,2] - delta
    max_z = location[0,2] + delta
    
    M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/cube/", device=device)
    
    M2 = dif.scale_cube_mesh(M2, 0.75)

    optimizer, scheduler = init_optimizer_scheduler([location])
    
#     with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
#         input_color = torch.load(f)

    # Read a PIL image
    image = PIL.Image.open('/home/ekim2/Storage/satellite-project/meshes/Adversarial-Texture/vehicle_camouflage.png')

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    adv_texture = img_tensor.permute(1,2,0) / 255

    lr_decay_rate = 2
    loss_fns = setup_loss_fns()

    iter_counter = 0
    image_id = 0
    num_cars = len(M)

    writer = SummaryWriter(log_dir=attack_logdir)

    blur_sigma = 0.7
    gaussian_blurrer = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=blur_sigma)
    
    directory = 'result_samples/cube-0.75_212-localized'
    
    os.makedirs(directory, exist_ok=True)
    os.makedirs(f'{directory}/location', exist_ok=True)

    print("Running the adversarial attack")

    with EventStorage(iter_counter) as storage:
        for epoch in range(5):
            progress_bar = tqdm(attack_loader, desc=f"Epoch #{epoch + 1}")
            for background_images_batch, labels_batch in progress_bar:
                batch_size = background_images_batch.shape[0]

                background_images_batch.to(device)
                
                random_sample_vehicle = random.randint(0, num_cars-1)

                ind = 0
                images_list = []
                
                images_input = dif.render_ontensor_continuous(batch_size, background_images_batch, M[random_sample_vehicle], M2, location, iter_counter)
                

                blur_images = blur_objects(gaussian_blurrer, images_input, background_images_batch)

                # #Debug
                if iter_counter%100 == 0:
                    img_counter = iter_counter
                    for syn_img in blur_images:
                        save_image(syn_img.detach().cpu(), f"./{directory}/imagetrain_{str(img_counter).zfill(5)}.png", quality=100)
                        img_counter += 1
                        
                model_inputs = construct_model_inputs(blur_images, batch_size, img_res, image_id)

                results = detector_attack_model(model_inputs)
                # print(results)

                loss_dict = losses_forward(results, model_inputs, loss_fns, [location])
                
                total_loss = sum(loss_dict.values()).to(device)
                
                # total_loss += location_loss * 1.5
                
#                 diff_comb_1 = torch.abs(torch.diff(torch.combinations(location[0]),dim=1))
#                 diff_comb_2 = torch.abs(torch.diff(torch.combinations(location[1]),dim=1))
                
#                 # diff_comb = torch.norm(diff_comb)
                
#                 location_difference_loss = torch.max(-(diff_comb_1 + diff_comb_2) + 1)
                
                # total_loss += location_difference_loss * 3
                
                total_loss.backward()
                
                sphere_vector_grad = torch.sum(torch.abs(location.grad))
                
                # with open(f'./{directory}/scale/iteration_{iter_counter:05d}.pkl','wb') as file:
                #     pickle.dump(scale,file)
                #     file.close()
                # with open(f'./{directory}/color/iteration_{iter_counter:05d}.pkl','wb') as file:
                #     pickle.dump(color,file)
                #     file.close()
                with open(f'./{directory}/location/iteration_{iter_counter:05d}.pkl','wb') as file:
                    pickle.dump(location,file)
                    file.close()
                    
                
                log_info(iter_counter, loss_dict, total_loss, results, writer, scheduler)
                log_params(iter_counter, sphere_vector_grad, writer)
                
                
                optimizer.step()
                optimizer.zero_grad()
                
                # location.data[0,1] = location.data[0,1].clamp(0,max_y+0.3)
                location.data[0,1] = location.data[0,1].clamp(min_y,max_y)
                location.data[0,0] = location.data[0,0].clamp(min_x,max_x)
                location.data[0,2] = location.data[0,2].clamp(min_z,max_z)

                iter_counter += 1

                #if(iter_counter % 100 == 0):
                #    break

            update_LR(scheduler, lr_decay_rate, epoch)
            
def attack_train_camera_ray(detector_attack_model, attack_loader, attack_logdir, device=None):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    img_res = 384


    dif = differentiablerenderer(device=device)

    M = dif.load_meshes(meshes_dir = "/home/ekim2/Storage/satellite-project/meshes/GAN-vehicles/", device=device)
    
    # set number of spheres
    num_spheres = 1

    with open('result_samples/cube-0.75_212/location/iteration_50639.pkl', 'rb') as f:
        location = pickle.load(f)
    location = location.to(device=device)
    
    dcamZ = init_adv(torch.zeros(1, device=device))
    
    
    M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/cube/", device=device)
    
    M2 = dif.scale_cube_mesh(M2, 0.75)

    optimizer, scheduler = init_optimizer_scheduler([dcamZ])

    lr_decay_rate = 2
    loss_fns = setup_loss_fns()

    iter_counter = 0
    image_id = 0
    num_cars = len(M)

    writer = SummaryWriter(log_dir=attack_logdir)

    blur_sigma = 0.7
    gaussian_blurrer = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=blur_sigma)
    
    directory = 'result_samples/cube-0.75_212-afterZ'
    
    os.makedirs(directory, exist_ok=True)
    os.makedirs(f'{directory}/dcamZ', exist_ok=True)

    print("Running the adversarial attack")

    with EventStorage(iter_counter) as storage:
        for epoch in range(5):
            progress_bar = tqdm(attack_loader, desc=f"Epoch #{epoch + 1}")
            for background_images_batch, labels_batch in progress_bar:
                batch_size = background_images_batch.shape[0]

                background_images_batch.to(device)
                
                random_sample_vehicle = random.randint(0, num_cars-1)

                ind = 0
                images_list = []
                
                images_input, location_loss = dif.render_ontensor_camera_ray(batch_size, background_images_batch, M[random_sample_vehicle], M2, location, iter_counter, dcamZ)
                

                blur_images = blur_objects(gaussian_blurrer, images_input, background_images_batch)

                # #Debug
                if iter_counter%100 == 0:
                    img_counter = iter_counter
                    for syn_img in blur_images:
                        save_image(syn_img.detach().cpu(), f"./{directory}/imagetrain_{str(img_counter).zfill(5)}.png", quality=100)
                        img_counter += 1
                        
                model_inputs = construct_model_inputs(blur_images, 4, img_res, image_id)

                results = detector_attack_model(model_inputs)

                loss_dict = losses_forward(results, model_inputs, loss_fns, [dcamZ])
                
                # total_loss = sum(loss_dict.values()).to(device)
                
                total_loss = location_loss
                
#                 diff_comb_1 = torch.abs(torch.diff(torch.combinations(location[0]),dim=1))
#                 diff_comb_2 = torch.abs(torch.diff(torch.combinations(location[1]),dim=1))
                
#                 # diff_comb = torch.norm(diff_comb)
                
#                 location_difference_loss = torch.max(-(diff_comb_1 + diff_comb_2) + 1)
                
                # total_loss += location_difference_loss * 3
                
                total_loss.backward()
                
                sphere_vector_grad = dcamZ.grad
                
                # with open(f'./{directory}/scale/iteration_{iter_counter:05d}.pkl','wb') as file:
                #     pickle.dump(scale,file)
                #     file.close()
                # with open(f'./{directory}/color/iteration_{iter_counter:05d}.pkl','wb') as file:
                #     pickle.dump(color,file)
                #     file.close()
                with open(f'./{directory}/dcamZ/iteration_{iter_counter:05d}.pkl','wb') as file:
                    pickle.dump(location,file)
                    file.close()
                    
                loss_dict['location loss'] = location_loss

                log_info(iter_counter, loss_dict, total_loss, results, writer, scheduler)
                log_params(iter_counter, sphere_vector_grad, writer)
                
                
                optimizer.step()
                optimizer.zero_grad()
                
                # location.data[0,1] = location.data[0,1].clamp(0,5)
                # location.data[0,0] = location.data[0,0].clamp(min_x,max_x)
                # location.data[0,2] = location.data[0,2].clamp(min_z,max_z)

                iter_counter += 1

                #if(iter_counter % 100 == 0):
                #    break

            update_LR(scheduler, lr_decay_rate, epoch)
            
def attack_train_cubeTexture(detector_attack_model, attack_loader, attack_logdir, device=None):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    img_res = 384

    dif = differentiablerenderer(device=device)

    M = dif.load_meshes(meshes_dir = "/home/ekim2/Storage/satellite-project/meshes/GAN-vehicles/", device=device)
    num_cars = len(M)
    
    # Read a PIL image
    image = PIL.Image.open('/home/ekim2/Storage/satellite-project/meshes/Adversarial-Texture/vehicle_camouflage.png')

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    adv_texture = img_tensor.permute(1,2,0) / 255
    
    init_vertices = [212, 120, 49, 50, 260, 113]
    mapping_vertex = np.load('/home/ekim2/Storage/satellite-project/meshes/004_281to962_vertex_indices_mapping.npy')[init_vertices]
    
    location = torch.nn.parameter.Parameter(init_adv(M[0].verts_packed()[mapping_vertex].clone().reshape(6,3)))
    
    with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
        input_color = torch.load(f).to(device)
    
    res = 32
    # set number of spheres
    # texture_map = init_adv(input_color[color_index].to(device) * torch.ones((320,320,3), device=device))
    texture_map = init_adv(torch.rand((res*res,5), device=device))    
    
    M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/cube/", device=device)
    
    M2 = dif.scale_cube_mesh(M2, 0.5)

    optimizer, scheduler = init_optimizer_scheduler([texture_map])

    lr_decay_rate = 2
    loss_fns = setup_loss_fns()

    iter_counter = 0
    image_id = 0

    writer = SummaryWriter(log_dir=attack_logdir)

    blur_sigma = 0.7
    gaussian_blurrer = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=blur_sigma)
    
    directory = 'result_plateTexture/cube-advTex'
    
    os.makedirs(directory, exist_ok=True)
    os.makedirs(f'{directory}/texture_map', exist_ok=True)
    
    print("Running the adversarial attack")

    with EventStorage(iter_counter) as storage:
        for epoch in range(5):
            progress_bar = tqdm(attack_loader, desc=f"Epoch #{epoch + 1}")
            for background_images_batch, labels_batch in progress_bar:
                batch_size = background_images_batch.shape[0]

                background_images_batch.to(device)

                ind = 0
                images_list = []
                
                random_sample_vehicle = random.randint(0, num_cars-1)
                
                images_input = dif.render_ontensor_optimizeCubeTexture(batch_size, background_images_batch, M[random_sample_vehicle], M2, texture_map, iter_counter, location, directory, adv_texture=adv_texture, input_color=input_color, res=res)
                

                blur_images = blur_objects(gaussian_blurrer, images_input, background_images_batch)

                # #Debug
                if iter_counter%100 == 0:
                    img_counter = iter_counter
                    for syn_img in blur_images:
                        save_image(syn_img.detach().cpu(), f"./{directory}/imagetrain_{str(img_counter).zfill(5)}.png", quality=100)
                        img_counter += 1
                        
                model_inputs = construct_model_inputs(blur_images, batch_size, img_res, image_id)

                results = detector_attack_model(model_inputs)

                loss_dict = losses_forward(results, model_inputs, loss_fns, [texture_map])
                
                total_loss = sum(loss_dict.values()).to(device)
                total_loss.backward()
                
                # texture_map_export = texture_map.clone().detach().cpu().permute(2,0,1)
                # save_image(texture_map,f'./{directory}/texture_map/iteration_{iter_counter:05d}.pkl')
                
                texture_grad = torch.sum(torch.abs(texture_map.grad))
                
                log_info(iter_counter, loss_dict, total_loss, results, writer, scheduler)
                log_params(iter_counter, texture_grad, writer)
                
                
                optimizer.step()
                optimizer.zero_grad()
                
                texture_map.data = texture_map.data.clamp(0,1)


                iter_counter += 1

                #if(iter_counter % 100 == 0):
                #    break

            update_LR(scheduler, lr_decay_rate, epoch)
            
            
def attack_train_cube_1(detector_attack_model, attack_loader, attack_logdir, device=None):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    img_res = 384
    
    vert_indices = [606,
                    489,
                    374,
                    542,
                    498,
                    427,
                    564,
                    490,
                    449,
                    579,
                    484,
                    421]


    dif = differentiablerenderer(device=device)

    M = dif.load_meshes(meshes_dir = "/home/ekim2/Storage/satellite-project/meshes/GAN-vehicles/", device=device)
    
    shape = 'cube'
    
    # set number of spheres
    num_spheres = 10
    
    silhouette_vector = torch.nn.parameter.Parameter(init_adv(0.5*torch.ones(num_spheres,device=device)))
    
    sphere_color = torch.nn.parameter.Parameter(init_adv(torch.rand([len(vert_indices),5],device=device)))
    
    # sphere_location = torch.nn.parameter.Parameter(init_adv(torch.rand([num_spheres, len(vert_indices)],device=device)))
    sphere_location = torch.nn.parameter.Parameter(init_adv(torch.rand([num_spheres, len(vert_indices)],device=device)))
    
    M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/cube/", device=device)
    
    M2 = dif.scale_cube_mesh(M2, 1.0)

    optimizer, scheduler = init_optimizer_scheduler([silhouette_vector, sphere_color, sphere_location])
    
    # with open('../background_sampled/cluster_5/sample_00000.pkl', 'rb') as f:
    #     input_color = pickle.load(f)
    
    with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
        input_color = torch.load(f)
    lr_decay_rate = 2
    loss_fns = setup_loss_fns()

    iter_counter = 0
    image_id = 0

    writer = SummaryWriter(log_dir=attack_logdir)

    blur_sigma = 0.7
    gaussian_blurrer = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=blur_sigma)
    
    directory = 'result_samples/results_ten_cube-1.0_LINZcolors-example'
    
    os.makedirs(directory, exist_ok=True)
    os.makedirs(f'{directory}/color', exist_ok=True)
    os.makedirs(f'{directory}/scale', exist_ok=True)
    os.makedirs(f'{directory}/location', exist_ok=True)

    print("Running the adversarial attack")

    with EventStorage(iter_counter) as storage:
        for epoch in range(5):
            progress_bar = tqdm(attack_loader, desc=f"Epoch #{epoch + 1}")
            for background_images_batch, labels_batch in progress_bar:
                batch_size = background_images_batch.shape[0]

                background_images_batch.to(device)

                ind = 0
                images_list = []
                
                images_input = dif.render_ontensor_limited_num(batch_size, background_images_batch, M[0], M2, silhouette_vector, sphere_color, sphere_location, iter_counter, input_color, vert_indices, shape)
                

                blur_images = blur_objects(gaussian_blurrer, images_input, background_images_batch)

                # #Debug
                if iter_counter%100 == 0:
                    img_counter = iter_counter
                    for syn_img in blur_images:
                        save_image(syn_img.detach().cpu(), f"./{directory}/imagetrain_{str(img_counter).zfill(5)}.png", quality=100)
                        img_counter += 1
                        
                model_inputs = construct_model_inputs(blur_images, 4, img_res, image_id)

                results = detector_attack_model(model_inputs)

                loss_dict = losses_forward(results, model_inputs, loss_fns, [silhouette_vector, sphere_color, sphere_location], sphere_location)
                

                total_loss = sum(loss_dict.values()).to(device)
                
                
                sphere_loss = soft_argmax(sphere_location)
                sphere_loss = torch.sum(sphere_loss, dim=0)
                sphere_loss = torch.abs(torch.max(sphere_loss) - 1)
                
                total_loss = total_loss + sphere_loss * 10
                
                
                total_loss.backward()
                
                sphere_vector_grad = torch.sum(torch.abs(sphere_location.grad))
                
                with open(f'./{directory}/scale/iteration_{iter_counter:05d}.pkl','wb') as file:
                    pickle.dump(silhouette_vector,file)
                    file.close()
                with open(f'./{directory}/color/iteration_{iter_counter:05d}.pkl','wb') as file:
                    pickle.dump(sphere_color,file)
                    file.close()
                with open(f'./{directory}/location/iteration_{iter_counter:05d}.pkl','wb') as file:
                    pickle.dump(sphere_location,file)
                    file.close()
                    
                
                log_info(iter_counter, loss_dict, total_loss, results, writer, scheduler)
                log_params(iter_counter, sphere_vector_grad, writer)
                
                
                optimizer.step()
                optimizer.zero_grad()
                
                silhouette_vector.data = silhouette_vector.data.clamp(0.4,1)

                iter_counter += 1

                #if(iter_counter % 100 == 0):
                #    break

            update_LR(scheduler, lr_decay_rate, epoch)


def generate_validation_data(attack_loader, device=None):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    img_res = 384


    with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
        input_color = torch.load(f)
        
    image = PIL.Image.open('/home/ekim2/Storage/satellite-project/meshes/Adversarial-Texture/vehicle_camouflage.png')

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    adv_texture = img_tensor.permute(1,2,0) / 255
    
       
    # silhouette_vector.clamp(0.01,1)
    save_dir = "/home/ekim2/Storage/satellite-project/Generated_Data/valdata_after-0.5_top6-bgColors-2-advTex/"
    set_type = "validation"

    scale = 0.5
    
#     # Read a PIL image
#     image = PIL.Image.open('result_plateTexture/example/texture_map/iteration_46274.png')
#     image = PIL.Image.open('result_plateTexture/example-adv/texture_map/iteration_90044.png')

#     # Define a transform to convert PIL 
#     # image to a Torch tensor
#     transform = transforms.Compose([
#         transforms.PILToTensor()
#     ])

#     # transform = transforms.PILToTensor()
#     # Convert the PIL image to Torch tensor
#     img_tensor = transform(image)
    
#     texture_map = img_tensor.permute(1,2,0) / 255
    
#     vert_index = 961
        
    dif = differentiablerenderer(device=device)

    M = dif.load_meshes(meshes_dir = "/home/ekim2/Storage/satellite-project/meshes/vehicles-masked/", device=device)
    num_cars = len(M)
    
    init_vertex = 212
    init_vertices = [212, 120, 49, 50, 260, 113]
    mapping_vertex = np.load('/home/ekim2/Storage/satellite-project/meshes/004_281to962_vertex_indices_mapping.npy')[init_vertices]
    
    location = torch.nn.parameter.Parameter(init_adv(M[0].verts_packed()[mapping_vertex].clone().reshape(6,3)))
#     with open('result_samples/cube-0.75_212-localized/location/iteration_39060.pkl', 'rb') as f:
#         location = pickle.load(f)
#     location = location.to(device=device)
    
    with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
        input_color = torch.load(f)
        
        
    color_index = 2
    
    M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/cube/", device=device)

    M2 = dif.scale_cube_mesh(M2, scale)

    iter_counter = 0
    image_id = 0

    blur_sigma = 0.7
    gaussian_blurrer = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=blur_sigma)

    print("Generating adverserial datasets")
    img_counter = 0

    with EventStorage(iter_counter) as storage:
        progress_bar = tqdm(attack_loader, desc=f"data")
        for background_images_batch, labels_batch in progress_bar:
            batch_size = background_images_batch.shape[0]

            background_images_batch.to(device)

            ind = 0
            images_list = []
            random_sample_vehicle = random.randint(0, num_cars-1)
            
            images_without_cube, images_with_cube, annotations = dif.render_ontensor_validation(batch_size, background_images_batch, M[random_sample_vehicle], M2, location, iter_counter, input_color=input_color[color_index])
            
            blur_images_without_cube = blur_objects(gaussian_blurrer, images_without_cube, background_images_batch)

            blur_images_with_cube = blur_objects(gaussian_blurrer, images_with_cube, background_images_batch)

            for k in range(batch_size):
                    # Adversarial image
                synthetic_image = blur_images_with_cube[k]
                PTH = os.path.join(save_dir, "adversarial", set_type, "images")
                image_save_path = os.path.join(save_dir, "adversarial", set_type, "images", f"image_{img_counter}_{k}.png")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                save_image(synthetic_image, image_save_path, quality=100)
                
                # Original image
                synthetic_image = blur_images_without_cube[k]
                PTH = os.path.join(save_dir, "original", set_type, "images")
                image_save_path = os.path.join(save_dir, "original", set_type, "images", f"image_{img_counter}_{k}.png")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                save_image(synthetic_image, image_save_path, quality=100)
                
                # Adversarial annotations
                anns_save_path = os.path.join(save_dir, "adversarial", set_type, "annotations", f"image_{img_counter}_{k}.pkl")
                PTH = os.path.join(save_dir, "adversarial", set_type, "annotations")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                with open(anns_save_path, 'wb') as f:
                    pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                # Original annotations
                anns_save_path = os.path.join(save_dir, "original", set_type, "annotations", f"image_{img_counter}_{k}.pkl")
                PTH = os.path.join(save_dir, "original", set_type, "annotations")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                with open(anns_save_path, 'wb') as f:
                    pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)

                img_counter += 1

            if img_counter >= 5000:
                break


            iter_counter += 1

def generate_validation_data1(attack_loader, device=None):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    img_res = 384


    with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
        input_color = torch.load(f)
        
    image = PIL.Image.open('/home/ekim2/Storage/satellite-project/meshes/vehicles-masked/masked_vehicle.png')

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    adv_texture = img_tensor.permute(1,2,0) / 255
    
       
    # silhouette_vector.clamp(0.01,1)
    save_dir = "/home/ekim2/Storage/satellite-project/Generated_Data/valdata_after-advtexture/"
    set_type = "validation"

    scale = 0.5
    
#     # Read a PIL image
#     image = PIL.Image.open('result_plateTexture/example/texture_map/iteration_46274.png')
#     image = PIL.Image.open('result_plateTexture/example-adv/texture_map/iteration_90044.png')

#     # Define a transform to convert PIL 
#     # image to a Torch tensor
#     transform = transforms.Compose([
#         transforms.PILToTensor()
#     ])

#     # transform = transforms.PILToTensor()
#     # Convert the PIL image to Torch tensor
#     img_tensor = transform(image)
    
#     texture_map = img_tensor.permute(1,2,0) / 255
    
#     vert_index = 961
        
    dif = differentiablerenderer(device=device)

    M = dif.load_meshes(meshes_dir = "/home/ekim2/Storage/satellite-project/meshes/GAN-vehicles/", device=device)
    num_cars = len(M)
    
    init_vertex = 212
    init_vertices = [212, 120, 49, 50, 260, 113]
    mapping_vertex = np.load('/home/ekim2/Storage/satellite-project/meshes/004_281to962_vertex_indices_mapping.npy')[init_vertices]
    
    location = torch.nn.parameter.Parameter(init_adv(M[0].verts_packed()[mapping_vertex].clone().reshape(6,3)))
#     with open('result_samples/cube-0.75_212-localized/location/iteration_39060.pkl', 'rb') as f:
#         location = pickle.load(f)
#     location = location.to(device=device)
    
    with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
        input_color = torch.load(f)
        
        
    color_index = 2
    
    M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/cube/", device=device)
    # M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/GAN-vehicles/", device=device)

    M2 = dif.scale_cube_mesh(M2, scale)

    iter_counter = 0
    image_id = 0

    blur_sigma = 0.7
    gaussian_blurrer = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=blur_sigma)

    print("Generating adverserial datasets")
    img_counter = 0

    with EventStorage(iter_counter) as storage:
        progress_bar = tqdm(attack_loader, desc=f"data")
        for background_images_batch, labels_batch in progress_bar:
            batch_size = background_images_batch.shape[0]

            background_images_batch.to(device)

            ind = 0
            images_list = []
            random_sample_vehicle = random.randint(0, num_cars-1)
            
            images_without_cube, images_with_cube, annotations = dif.render_ontensor_validation1(batch_size, background_images_batch, M[random_sample_vehicle], M2, location, iter_counter, adv_texture=adv_texture)
            
            blur_images_without_cube = blur_objects(gaussian_blurrer, images_without_cube, background_images_batch)

            blur_images_with_cube = blur_objects(gaussian_blurrer, images_with_cube, background_images_batch)

            for k in range(batch_size):
                    # Adversarial image
                synthetic_image = blur_images_with_cube[k]
                PTH = os.path.join(save_dir, "adversarial", set_type, "images")
                image_save_path = os.path.join(save_dir, "adversarial", set_type, "images", f"image_{img_counter}_{k}.png")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                save_image(synthetic_image, image_save_path, quality=100)
                
                # Original image
                synthetic_image = blur_images_without_cube[k]
                PTH = os.path.join(save_dir, "original", set_type, "images")
                image_save_path = os.path.join(save_dir, "original", set_type, "images", f"image_{img_counter}_{k}.png")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                save_image(synthetic_image, image_save_path, quality=100)
                
                # Adversarial annotations
                anns_save_path = os.path.join(save_dir, "adversarial", set_type, "annotations", f"image_{img_counter}_{k}.pkl")
                PTH = os.path.join(save_dir, "adversarial", set_type, "annotations")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                with open(anns_save_path, 'wb') as f:
                    pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                # Original annotations
                anns_save_path = os.path.join(save_dir, "original", set_type, "annotations", f"image_{img_counter}_{k}.pkl")
                PTH = os.path.join(save_dir, "original", set_type, "annotations")
                isExist = os.path.exists(PTH)
                if not isExist:
                    os.makedirs(PTH)
                with open(anns_save_path, 'wb') as f:
                    pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)

                img_counter += 1

            if img_counter >= 5000:
                break


            iter_counter += 1


            
            
def generate_SimpleAttackforHeatMap(attack_loader, device=None):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    img_res = 384

    with open('../background_sampled/LINZ/camouflage_colors.pth', 'rb') as f:
        input_color = torch.load(f)

    save_dir = "/home/ekim2/Storage/satellite-project/HeatMap-Examples-0.75-2cubes/"
    set_type = "validation"

    scale = 0.75
    
    dif = differentiablerenderer(device=device)

    M = dif.load_meshes(meshes_dir = "/home/ekim2/Storage/satellite-project/meshes/simplified_cars_281verts/", device=device)
    
    M2 = dif.load_meshes(meshes_dir="/home/ekim2/Storage/satellite-project/meshes/silhouette_camouflage/cube/", device=device)

    M2 = dif.scale_cube_mesh(M2, scale)

    iter_counter = 0
    image_id = 0

    blur_sigma = 0.7
    gaussian_blurrer = torchvision.transforms.GaussianBlur(kernel_size = 5, sigma=blur_sigma)

    print("Generating adverserial datasets")
    img_counter = 0

    with EventStorage(iter_counter) as storage:
        progress_bar = tqdm(attack_loader, desc=f"data")
        for background_images_batch, labels_batch in progress_bar:
            batch_size = background_images_batch.shape[0]

            background_images_batch.to(device)

            ind = 0
            images_list = []
            
            images_with_cube, annotations = dif.render_ontensor_evaluateVertexAttackHeatMap(batch_size, background_images_batch, M[-1], M2, iter_counter, 'cube')

            blur_images_with_cube_l = [blur_objects(gaussian_blurrer, img, background_images_batch) for img in images_with_cube]
            
            for vertex_number, blur_images_with_cube in enumerate(blur_images_with_cube_l):
                
                for k in range(batch_size):
                    # Adversarial image
                    synthetic_image = blur_images_with_cube[k]
                    PTH = os.path.join(save_dir, "adversarial", set_type, "images")
                    image_save_path = os.path.join(save_dir, "adversarial", set_type, "images", f"image_{vertex_number:03d}_{iter_counter:06d}_{k}.png")
                    isExist = os.path.exists(PTH)
                    if not isExist:
                        os.makedirs(PTH)
                    save_image(synthetic_image, image_save_path, quality=100)

                    # Adversarial annotations
                    anns_save_path = os.path.join(save_dir, "adversarial", set_type, "annotations", f"image_{vertex_number:03d}_{iter_counter:06d}_{k}.pkl")
                    PTH = os.path.join(save_dir, "adversarial", set_type, "annotations")
                    isExist = os.path.exists(PTH)
                    if not isExist:
                        os.makedirs(PTH)
                    with open(anns_save_path, 'wb') as f:
                        pickle.dump(annotations[k], f, protocol=pickle.HIGHEST_PROTOCOL)

                img_counter += 1

            if iter_counter >= 25:
                break


            iter_counter += 1
            


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

if __name__ == '__main__':
    #Load the attacker model
    batch_size = 8
    detector_attack_config = "./RetinaConfig.yaml"
    attack_background_data = "/home/ekim2/Storage/satellite-project/SatDet-Real-GoogleMaps-384px-0.125m/"
    device = torch.device('cuda:1')
    attack_logdir = "./attack_output_generator/cube-advTex/"
    
    attack_train = True
    
    if attack_train:
        cfg = get_cfg()
        cfg.merge_from_file(detector_attack_config)
        cfg.freeze()
        args = None
        default_setup(
            cfg, args
        )  # if you don't like any of the default setup, write your own setup code
        detector_attack_model = build_model(cfg)
        DetectionCheckpointer(detector_attack_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                    cfg.MODEL.WEIGHTS, resume=True
                )
        freeze_model(detector_attack_model)
        augmentations = T.AugmentationList([
            T.RandomContrast(0.2, 2.0),
            T.RandomBrightness(0.2, 2.0),
            T.RandomSaturation(0.2, 2.0),
            T.RandomLighting(1.0),
            T.RandomFlip(horizontal=True),
            T.RandomFlip(horizontal=False, vertical=True),
            T.RandomRotation([0.0, 360.0], expand=False)
        ])
    else:
        augmentations = None

    print('==> Attacking train')

    if attack_train:
        attack_set = AttackDataset(attack_background_data, augmentations=augmentations, device=device)
        attack_loader = DataLoader(attack_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        attack_train_cubeTexture(
        detector_attack_model, attack_loader, attack_logdir, device=device
        )

    generate_data = False
    if generate_data:
        attack_background_data = "../SatDet-Real-GoogleMaps-384px-0.125m/val/"
        attack_set = AttackDataset(attack_background_data, augmentations=augmentations, device=device)
        attack_loader = DataLoader(attack_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        generate_validation_data1(attack_loader, device=device)

    heatmap_data = False
    if heatmap_data:
        attack_background_data = "../SatDet-Real-GoogleMaps-384px-0.125m/val/"
        attack_set = AttackDataset(attack_background_data, augmentations=augmentations, device=device)
        attack_loader = DataLoader(attack_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        generate_SimpleAttackforHeatMap(attack_loader,device=device)