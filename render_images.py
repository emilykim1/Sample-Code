import os
import glob
import math
import torch
import torchvision
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from functools import reduce
from torchvision import transforms
#from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch3d.io import load_objs_as_meshes, save_obj, IO
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix
from torch.utils.data.distributed import DistributedSampler
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, FoVOrthographicCameras, look_at_view_transform
from pytorch3d.transforms import Rotate
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import point_mesh_face_distance
import PIL.Image
from torch import nn
from torchvision.transforms import Resize

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    BlendParams
)
from pytorch3d.structures import join_meshes_as_batch, Meshes

import pdb

import matplotlib.pyplot as plt

#from attacker.UniformTexturesAttacker import UniformTexturesAttacker

class differentiablerenderer():
    def __init__(self, device) -> None:
        self.DISTANCE = 5.0
        self.SCALING_FACTORS_RANGE = [0.10, 0.12]
        self.INTENSITIES_RANGE = [0.5, 2.0]
        self.device = device
        pass

    def load_meshes(self, meshes_dir, device = 'cpu'):
        meshes = []
        obj_paths = glob.glob(meshes_dir + "*.obj")
        for obj_path in tqdm(obj_paths):
            mesh = load_objs_as_meshes([obj_path], device=device)[0]
            meshes.append(mesh)
        # mesh = load_objs_as_meshes([obj_paths[0]], device=device)[0]
        # print(mesh.shape)
        return meshes
    
    def extract_screen_coordinates(self, meshes, distances, elevations, azimuths, scaling_factors):
        assert len(meshes) == len(distances) == len(elevations) == len(azimuths)
        image_size = 384
        
        # Initialize the cameras
        scaling_factors = scaling_factors.repeat(1, 3)
        R, T = look_at_view_transform(dist=distances, elev=elevations, azim=azimuths)
        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            scale_xyz=scaling_factors
        )
        batch_size = len(meshes)
        locations_batch = []
        for i in range(batch_size):
            # Extract locations for the given image
            meshes_ = meshes[i]
            verts = torch.stack([mesh.verts_padded() for mesh in meshes_]).squeeze(1)
            center_coords = torch.mean(verts, dim=1)
            locations = cameras.transform_points_screen(center_coords, image_size=(image_size, image_size))[i, :, :2]
            
            # Remove locations which fall outside the image
            valid_list = []
            for idx, location in enumerate(locations):
                if 0 <= location[0] <= image_size and 0 <= location[1] <= image_size:
                    valid_list.append(idx)
            locations = locations[[valid_list]]
            
            # Append to the batch
            locations_batch.append(locations)
        return locations_batch
    

    def randomly_move_and_rotate_mesh(self, mesh, scaling_factor):
        # Apply random rotation
        mesh_rotation1 = euler_angles_to_matrix(torch.tensor([0, 0, 0]), convention="XYZ").to(self.device)
        mesh_rotation = torch.matmul(mesh_rotation1, mesh.verts_packed().data.T).T - mesh.verts_packed()
        mesh.offset_verts_(vert_offsets_packed=mesh_rotation)
        
        # Apply random translation (forcing the center of the vehicle to stay in the image)
        mesh_dx = random.uniform(-0.8, 0.8)
        mesh_dz = random.uniform(-0.8, 0.8)
        mesh_dy = torch.max(mesh.verts_list()[0], 0).values[1].item()
        # Compute the offset
        
        
        mesh_dx /= scaling_factor
        mesh_dz /= scaling_factor
        mesh_dy /= scaling_factor

        
        
        # Center the mesh before applying translation
        mesh_dx -= torch.mean(mesh.verts_padded(), dim=1)[0][0].item()
        mesh_dz -= torch.mean(mesh.verts_padded(), dim=1)[0][2].item()
        offset = torch.cat([-mesh_dx.cpu().detach(), mesh_dy.cpu().detach(), -mesh_dz.cpu().detach()]) # To be in the (x, y) format on the image
        
        # Apply the translation
        mesh_translation = torch.tensor([mesh_dx, 0, mesh_dz], device=self.device) * torch.ones(size=mesh.verts_padded().shape[1:], device=self.device)
        mesh.offset_verts_(vert_offsets_packed=mesh_translation)
        
        return (mesh.clone(), offset, mesh_rotation1)
    
    def randomly_move_and_rotate_mesh_cube(self, mesh, scaling_factor, offset, mesh_rotation):
        #print(len(mesh_rotation))
        # Apply random rotation
        #mesh_rotation = euler_angles_to_matrix(torch.tensor([0, random.uniform(0, 2 * math.pi), 0]), convention="XYZ").to(self.device)
        mesh_rotation = torch.matmul(mesh_rotation[0], mesh.verts_packed().data.T).T - mesh.verts_packed()
        mesh.offset_verts_(vert_offsets_packed=mesh_rotation)
        
        # Apply random translation (forcing the center of the vehicle to stay in the image)
        #mesh_dx = random.uniform(-1, 1)
        #mesh_dz = random.uniform(-1, 1)
        
        # Compute the offset
        #offset = np.array([-mesh_dx, -mesh_dz]) # To be in the (x, y) format on the image
        mesh_dx = -offset[0][0]
        mesh_dz = -offset[0][2]
        mesh_dy = offset[0][1]
        

        print(mesh_dx, mesh_dy, mesh_dz)

        
        
        # Center the mesh before applying translation
        #mesh_dx -= torch.mean(mesh.verts_padded(), dim=1)[0][0].item()
        #mesh_dz -= torch.mean(mesh.verts_padded(), dim=1)[0][2].item()
        
        # Apply the translation
        mesh_translation = torch.tensor([mesh_dx, mesh_dy, mesh_dz], device=self.device) * torch.ones(size=mesh.verts_padded().shape[1:], device=self.device)
        mesh.offset_verts_(vert_offsets_packed=mesh_translation)
        
        return (mesh.clone(), offset)
    
    def randomly_move_and_rotate_meshes(self, meshes, scaling_factor, distance, elevation, azimuth, intensity):
        invalid_image = True
        
        # Create the silhouette renderer
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        cameras = FoVOrthographicCameras(
            device=self.device, 
            R=R,
            T=T, 
            scale_xyz=((scaling_factor, scaling_factor, scaling_factor),)
        ) 
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=384, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=1, 
            bin_size=-1
        )
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        
        while invalid_image:
            offsets = []
            silhouettes = []
            mesh_rotations = []
            
            for i in range(len(meshes)):
                meshes[i], offset, mesh_rotation = self.randomly_move_and_rotate_mesh(meshes[i], scaling_factor)
                silhouette = silhouette_renderer(meshes[i], cameras=cameras)
                silhouette = (silhouette[..., 3] > 0.5).float()
                silhouettes.append(silhouette)
                offsets.append(offset)
                mesh_rotations.append(mesh_rotation)
            
            # Check whether any of the meshes intersect
            if torch.any(reduce(lambda x, y: x + y, silhouettes) > 1.0):
                invalid_image = True
            else:
                invalid_image = False
        
        return (meshes, offsets, mesh_rotations)
    
    def randomly_move_and_rotate_meshes_cube(self, meshes, scaling_factor, distance, elevation, azimuth, intensity, offset, rotation):
        invalid_image = True
        
        # Create the silhouette renderer
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        cameras = FoVOrthographicCameras(
            device=self.device, 
            R=R,
            T=T, 
            scale_xyz=((scaling_factor, scaling_factor, scaling_factor),)
        ) 
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=384, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
            bin_size=-1
        )
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        
        while invalid_image:
            offsets = []
            silhouettes = []
            
            for i in range(len(meshes)):
                meshes[i], offset = self.randomly_move_and_rotate_mesh_cube(meshes[i], scaling_factor, offset, rotation)
                silhouette = silhouette_renderer(meshes[i], cameras=cameras)
                silhouette = (silhouette[..., 3] > 0.5).float()
                silhouettes.append(silhouette)
                offsets.append(offset)
            
            # Check whether any of the meshes intersect
            if torch.any(reduce(lambda x, y: x + y, silhouettes) > 1.0):
                invalid_image = True
            else:
                invalid_image = False
        
        return (meshes, offsets)
    
    def randomly_place_meshes(self, meshes, scaling_factors):
        offsets = [None for _ in range(len(meshes))]
        for i in range(len(meshes)):
            meshes[i], offsets[i] = self.randomly_move_and_rotate_mesh(meshes[i], scaling_factors[i])
        return meshes, offsets
    
    # TO DO: move this function to utils
    def randomly_place_meshes_multi(self, meshes_list, scaling_factors, distances, elevations, azimuths, intensities):
        assert len(meshes_list) == len(scaling_factors)
        meshes = []
        offsets = []
        rotations = []
        
        for i in range(len(meshes_list)):
            meshes_ = meshes_list[i]
            scaling_factor = scaling_factors[i]
            distance = distances[i]
            elevation = elevations[i]
            azimuth = azimuths[i]
            intensity = intensities[i]
            
            meshes_, offsets_, rotations_ = self.randomly_move_and_rotate_meshes(meshes_, scaling_factor, distance, elevation, azimuth, intensity)
            
            meshes.append(meshes_)
            offsets.append(offsets_)
            rotations.append(rotations_)
            
        
        locations_batch = self.extract_screen_coordinates(meshes, distances, elevations, azimuths, scaling_factors)
        
        return meshes, locations_batch, offsets, rotations
    
    def randomly_place_meshes_multi_cube(self, meshes_list, scaling_factors, distances, elevations, azimuths, intensities, offsets, rotations):
        assert len(meshes_list) == len(scaling_factors)
        meshes = []
        offsets_e = []
        
        for i in range(len(meshes_list)):
            meshes_ = meshes_list[i]
            scaling_factor = scaling_factors[i]
            distance = distances[i]
            elevation = elevations[i]
            azimuth = azimuths[i]
            intensity = intensities[i]
            offset = offsets[i]
            rotation = rotations[i]
            
            meshes_, offsets_ = self.randomly_move_and_rotate_meshes_cube(meshes_, scaling_factor, distance, elevation, azimuth, intensity, offset, rotation)
            
            meshes.append(meshes_)
            offsets_e.append(offsets_)
            
        
        locations_batch = self.extract_screen_coordinates(meshes, distances, elevations, azimuths, scaling_factors)
        
        return meshes, locations_batch


        
    def sample_random_elev_azimuth(self, x_min, y_min, x_max, y_max, distance):
        """
        This function samples x and y coordinates on a plane, and converts them to elevation and azimuth angles.
        
        It was found that x_min = y_min = -1.287 and x_max = y_max = 1.287 result in the best angles, where elevation ranges roughly from 70 to 90, and azimuth goes from 0 to 360.
        """
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        if x == 0 and y == 0:
            elevation = 90.0
            azimuth = 0.0
        elif x == 0:
            elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi
            azimuth = 0.0
        else:
            elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi 
            azimuth = math.atan(y / x) * 180.0 / math.pi
            if x < 0:
                if y > 0:
                    azimuth += 180
                else:
                    azimuth -= 180

        return (elevation, azimuth)


    def sample_rendering_params_close_up(self, batch_size):
        distance = 2
        # elevation = self.attack_cfg.RENDERER.ELEVATION
        # azimuth = self.attack_cfg.RENDERER.AZIMUTH
        sf_range = [3.0, 3.2]
        int_range = self.INTENSITIES_RANGE
        els_azs = [(0, 0) for _ in range(batch_size)]
        distances = [distance for _ in range(batch_size)]
        # elevations = [elevation for _ in range(batch_size)]
        # azimuths = [azimuth for _ in range(batch_size)]
        elevations = [els_azs_[0] for els_azs_ in els_azs]
        azimuths = [els_azs_[1] for els_azs_ in els_azs] # No need to rotate the camera if the vehicles is rotated
        lights_directions = torch.rand(batch_size, 3, device=self.device) * 2 - 1
        lights_directions[:, 1] = -1
        scaling_factors = torch.ones(batch_size, 1, device=self.device) * (sf_range[1] - sf_range[0]) + sf_range[0]
        intensities = torch.ones(batch_size, 1, device=self.device) * (int_range[1] - int_range[0]) + int_range[0]
        
        return (distances, elevations, azimuths, lights_directions, scaling_factors, intensities)
    
    def sample_rendering_params(self, batch_size):
        distance = self.DISTANCE
        # elevation = self.attack_cfg.RENDERER.ELEVATION
        # azimuth = self.attack_cfg.RENDERER.AZIMUTH
        sf_range = self.SCALING_FACTORS_RANGE
        int_range = self.INTENSITIES_RANGE
        els_azs = [self.sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, self.DISTANCE) for _ in range(batch_size)]
        distances = [distance for _ in range(batch_size)]
        # elevations = [elevation for _ in range(batch_size)]
        # azimuths = [azimuth for _ in range(batch_size)]
        elevations = [els_azs_[0] for els_azs_ in els_azs]
        azimuths = [els_azs_[1] for els_azs_ in els_azs] # No need to rotate the camera if the vehicles is rotated
        lights_directions = torch.rand(batch_size, 3, device=self.device) * 2 - 1
        lights_directions[:, 1] = -1
        scaling_factors = torch.rand(batch_size, 1, device=self.device) * (sf_range[1] - sf_range[0]) + sf_range[0]
        intensities = torch.rand(batch_size, 1, device=self.device) * (int_range[1] - int_range[0]) + int_range[0]
        
        return (distances, elevations, azimuths, lights_directions, scaling_factors, intensities)
    
    def camera_position_from_spherical_angles(self, distance, elevation, azimuth, degrees=True):
        dist, elev, azim = torch.tensor(distance, device = self.device).unsqueeze(0), torch.tensor(elevation, device = self.device).unsqueeze(0), torch.tensor(azimuth, device = self.device).unsqueeze(0)
        if degrees:
            elev = math.pi / 180.0 * elev
            azim = math.pi / 180.0 * azim
        x = dist * torch.cos(elev) * torch.sin(azim)
        y = dist * torch.sin(elev)
        z = dist * torch.cos(elev) * torch.cos(azim)
        camera_position = torch.stack([x, y, z], dim=1)
        if camera_position.dim() == 0:
            camera_position = camera_position.view(1, -1)  # add batch dim.
        return camera_position.view(-1, 3)
    
    def construct_annotations_files(self, locations_batch):
        annotations_batch = []
        
        for locations in locations_batch:
            annotations = {
                "van_rv": np.empty(shape=(0, 2)),
                "truck": np.empty(shape=(0, 2)),
                "bus": np.empty(shape=(0, 2)),
                "trailer_small": np.empty(shape=(0, 2)),
                "specialized": np.empty(shape=(0, 2)),
                "trailer_large": np.empty(shape=(0, 2)),
                "unknown": np.empty(shape=(0, 2)),
                "small": locations.detach().cpu().numpy()
            }
            annotations_batch.append(annotations)
    
        return annotations_batch
    
    def render_batch(self, meshes, background_images, elevations, azimuths, light_directions, distances,
                     scaling_factors, intensities, image_size=384, blur_radius=0.0, faces_per_pixel=1, ambient_color=((0.05, 0.05, 0.05),)):
        # Image needs to be upscaled and then average pooled to make the car less sharp-edged
        transform = Resize((image_size, image_size))
        background_images = transform(background_images).permute(0, 2, 3, 1)
        scaling_factors = scaling_factors.repeat(1, 3)
        intensities = intensities.repeat(1, 3)

        
        R, T = look_at_view_transform(dist=distances, elev=elevations, azim=azimuths)

        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            scale_xyz=scaling_factors
        )
        
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=blur_radius, 
            faces_per_pixel=faces_per_pixel, 
            bin_size=-1
        )
        
        lights = DirectionalLights(
            device=self.device, 
            direction=light_directions, 
            ambient_color=ambient_color, 
            diffuse_color=intensities
        )
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=background_images)
            )
        )
        
        if isinstance(meshes, Meshes):
            pass
        elif isinstance(meshes, list):
            meshes = join_meshes_as_batch(meshes)
        else:
            raise Exception("Incorrect data type for the 'meshes' variable.")
        
        images = renderer(meshes, lights=lights, cameras=cameras)
        images = images[..., :3]
        images = images.permute(0, 3, 1, 2)
        
        return images
    
    def transform_cube_mesh(self, mesh_cube, mesh_car):
        # print(width, length)
        # inds = mesh_cube[0].verts_list()[0][:, 0] == 0.3
        # deform_verts = torch.full(mesh_cube[0].verts_packed().shape, 0.0, device=self.device)
        # deform_verts[inds, 0] = 0.3*width
        # mesh_cube[0] = mesh_cube[0].offset_verts(deform_verts)

        # inds = mesh_cube[0].verts_list()[0][:, 2] == 0.3
        # deform_verts = torch.full(mesh_cube[0].verts_packed().shape, 0.0, device=self.device)
        # deform_verts[inds, 2] = 0.3*length
        # mesh_cube[0] = mesh_cube[0].offset_verts(deform_verts)
        #print(scaling_factor)
        mesh_dx = torch.mean(mesh_car[0].verts_padded(), dim=1)[0][0].item() - torch.mean(mesh_cube[0].verts_padded(), dim=1)[0][0].item()
        mesh_dz = torch.mean(mesh_car[0].verts_padded(), dim=1)[0][2].item() - torch.mean(mesh_cube[0].verts_padded(), dim=1)[0][2].item()
        mesh_dy = torch.max(mesh_car[0].verts_list()[0], 0).values[1].item() #+ torch.max(mesh_cube[0].verts_list()[0], 0).values[1].item()
        mesh_translation = torch.tensor([mesh_dx, mesh_dy, mesh_dz], device=self.device) * torch.ones(size=mesh_cube[0].verts_padded().shape[1:], device=self.device)
        mesh_cube[0].offset_verts_(vert_offsets_packed=mesh_translation)

        return mesh_cube[0]
    
    def scale_cube_mesh(self, mesh_cube, scale):
        mesh_cube[0].scale_verts_(scale)
        # deform_verts = torch.full(mesh_cube[0].verts_packed().shape, 0.0, device=self.device)
        # inds = mesh_cube[0].verts_list()[0][:, 1] == -0.3
        # deform_verts[inds, 1] += 0.15
        # mesh_cube[0] = mesh_cube[0].offset_verts(deform_verts)

        return mesh_cube[0]
    
     
    def save_textures(self, adv_textures, epoch, final=False):
        # Extract the textures from the texture generator
        textures = adv_textures
        core_textures = adv_textures
        # Save path
        if final:
            save_path_image = os.path.join("./results_texwlc_cons_2/", "adv_textures_final.png")
            save_path_tensor = os.path.join("./results_texwlc_cons_2/", "adv_textures_final.pth")
        else:
            save_path_image = os.path.join("./results_texwlc_cons_2/", f"adv_textures_{epoch}.png")
            save_path_tensor = os.path.join("./results_texwlc_cons_2/", f"adv_textures_{epoch}.pth")
        
        # Save as an image (PNG)
        save_image(textures.permute(2, 0, 1), save_path_image, quality=100)
        
        # Save as a torch.Tensor (PTH)
        torch.save(core_textures, save_path_tensor)
    
    def join_meshes_and_replace_textures_1(self, meshes_batch, batch_size, silhouette_vector, sphere_location, sphere_color, input_color, num_vehicle_vertices, num_sphere_vertices, vert_indices, shape='sphere'):
        meshes_batch = join_meshes_as_batch(meshes_batch)
        
        sphere_loc = torch.zeros_like(sphere_location)
        
        for i in range(sphere_location.shape[0]):
            sphere_loc[[i]] = self.soft_argmax(sphere_location[[i],:])

        sphere_location = sphere_loc
                
        deform_vector = torch.zeros_like(sphere_location[0],device=self.device)
                
        for i in range(sphere_location.shape[0]):
            deform_vector += sphere_location[i] * silhouette_vector[i]
            
        # deform_vector = deform_vector / torch.norm(deform_vector)
        deform_vector = torch.clamp(deform_vector,min=0,max=1)
        
        deform_verts3 = torch.full(meshes_batch[0].verts_packed().shape, 0.0, device=self.device)

        centers = []
        
        for i in range(len(vert_indices)):
            # center = torch.mean(meshes_batch.verts_packed()[num_vehicle_vertices + i * num_sphere_vertices:num_vehicle_vertices + (i+1) * num_sphere_vertices,],dim=0)
            center = meshes_batch.verts_packed()[vert_indices[i]]
            centers += [center.repeat(num_sphere_vertices,1)]
        centers = torch.cat(centers, dim=0)
        

        # normals = meshes_batch[0].verts_normals_packed()[:num_vehicle_vertices]  # Needs fix
        
        displacement = -(meshes_batch[0].verts_packed()[num_vehicle_vertices:,] - centers) * (1-deform_vector.reshape(-1,1).repeat(1, num_sphere_vertices).reshape(-1,1))
        
        # print(displacement.shape)
        deform_verts3[num_vehicle_vertices:,] = displacement
        meshes_batch.offset_verts_(vert_offsets_packed=deform_verts3.repeat(batch_size, 1))
        
        if shape == 'sphere':
            normals = meshes_batch[0].verts_normals_packed()[vert_indices]

            normals = normals.repeat(1,num_sphere_vertices).reshape(-1,3)
            radius = torch.norm(meshes_batch[0].verts_packed()[num_vehicle_vertices:,] - centers, dim=1)    
            # print(radius.shape)
            # print(normals.shape)

            displacement_1 = - normals * radius.reshape(-1,1)

            deform_verts3[num_vehicle_vertices:,] = displacement_1
            meshes_batch.offset_verts_(vert_offsets_packed=deform_verts3.repeat(batch_size, 1))
        
        if shape == 'cube':
            normals = meshes_batch[0].verts_normals_packed()[vert_indices]

            normals = (normals/torch.norm(normals, dim=0)).repeat(1,num_sphere_vertices).reshape(-1,3)
            radius = torch.norm(meshes_batch[0].verts_packed()[num_vehicle_vertices:,] - centers, dim=1) / (torch.sqrt(torch.ones(meshes_batch[0].verts_packed()[num_vehicle_vertices:,].shape[0], device=self.device)*2))
            
            
#             for i in range(len(radius)//8):
#                 print(radius[i * 8:8*i + 8])
            


            displacement_1 = - normals * radius.reshape(-1,1)

            deform_verts3[num_vehicle_vertices:,] = displacement_1
            meshes_batch.offset_verts_(vert_offsets_packed=deform_verts3.repeat(batch_size, 1))
            
        
        ## New Sphere texture

        sorted_centroids = torch.tensor(input_color,device=self.device)
        

        width, length = meshes_batch.textures._maps_padded[:,:,512:,:].shape[1:3]

        color = self.soft_argmax(sphere_color)
        color = color.double() @ sorted_centroids.double()

        
        sphere_counter = 0


        for j in range(length//100): # length
            for i in range(width//100): # width
                meshes_batch.textures._maps_padded[:,100 * i: 100 * (i + 1), 512 + 100 * j:512 + 100 * (j+1),:] = color[sphere_counter].reshape(1,3)

                sphere_counter += 1

                if sphere_counter == len(vert_indices):
                    break
            if sphere_counter == len(vert_indices):
                    break
        
        return meshes_batch
    
    def join_meshes_and_replace_textures(self, meshes_batch, batch_size, sphere_location, num_vehicle_vertices, num_sphere_vertices, adv_texture=None, input_color=None):
        meshes_batch = join_meshes_as_batch(meshes_batch)

        # First move the object then scale!
        
        deform_verts3_l = []
        num_spheres = sphere_location.shape[0]
        sphere_location = sphere_location.repeat(1, num_sphere_vertices).reshape(num_sphere_vertices * sphere_location.shape[0], 3)
        for j in range(batch_size):
            centers = []
            for i in range(num_spheres):
                # object center
                center = torch.mean(meshes_batch[j].verts_packed()[num_vehicle_vertices + i * num_sphere_vertices : num_vehicle_vertices + (i+1) * num_sphere_vertices], dim=0)
                centers += [center.repeat(num_sphere_vertices,1)]
            centers = torch.cat(centers,dim=0)
            
            # vehicle center
            center_v = torch.mean(meshes_batch[j].verts_packed()[:num_vehicle_vertices], dim=0)
            center_v = center_v.repeat(meshes_batch[j].verts_packed()[num_vehicle_vertices:].shape[0],1)
            
            deform_verts3 = torch.full(meshes_batch[0].verts_packed().shape, 0.0, device=self.device)

            offset_loc = sphere_location - centers + center_v

            deform_verts3[num_vehicle_vertices:] = offset_loc
            
            deform_verts3_l += [deform_verts3]
            
        deform_verts3_l = torch.cat(deform_verts3_l, dim=0)
        meshes_batch.offset_verts_(vert_offsets_packed=deform_verts3_l)
        
        if adv_texture != None:
            meshes_batch.textures._maps_padded[:,:512,:512,:] = adv_texture
        if input_color != None:
            meshes_batch.textures._maps_padded[:,:,512:,:] = input_color
        
        # scale
#         deform_verts3_s = []
#         for j in range(batch_size):
#             centers = []
            
#             for i in range(num_spheres):
#                 # object center
#                 center = torch.mean(meshes_batch[j].verts_packed()[num_vehicle_vertices + i * num_sphere_vertices : num_vehicle_vertices + (i+1) * num_sphere_vertices], dim=0)
#                 centers += [center.repeat(num_sphere_vertices,1)]
#             centers = torch.cat(centers,dim=0)
            
#             deform_verts3 = torch.full(meshes_batch[0].verts_packed().shape, 0.0, device=self.device)

#             displacement = -(meshes_batch[j].verts_packed()[num_vehicle_vertices:,] - centers) * (1-deform_vector.reshape(-1,1).repeat(1, num_sphere_vertices).reshape(-1,1))

#             deform_verts3[num_vehicle_vertices:,] = displacement
            
#             deform_verts3_s += [deform_verts3]
            
#         deform_verts3_s = torch.cat(deform_verts3_s, dim=0)
#         meshes_batch.offset_verts_(vert_offsets_packed=deform_verts3_s)

        ## New Sphere texture

#         sorted_centroids = torch.tensor(input_color,device=self.device)
        

#         width, length = meshes_batch.textures._maps_padded[:,:,512:,:].shape[1:3]

#         color = self.soft_argmax(sphere_color)
#         color = color.double() @ sorted_centroids.double()

#         sphere_counter = 0


#         for j in range(length//100): # length
#             for i in range(width//100): # width
#                 meshes_batch.textures._maps_padded[:,100 * i: 100 * (i + 1), 512 + 100 * j:512 + 100 * (j+1),:] = color[sphere_counter].reshape(1,3)

#                 sphere_counter += 1

#                 if sphere_counter == num_spheres:
#                     break
#             if sphere_counter == num_spheres:
#                     break
        
        return meshes_batch
    
    def join_meshes_and_replace_cube_textures(self, meshes_batch, batch_size, sphere_location, texture_map, num_vehicle_vertices, num_sphere_vertices, adv_texture=None, input_color=None, res=16):
        meshes_batch = join_meshes_as_batch(meshes_batch)
        
        deform_verts3_l = []
        num_spheres = sphere_location.shape[0]
        sphere_location = sphere_location.repeat(1, num_sphere_vertices).reshape(num_sphere_vertices * sphere_location.shape[0], 3)
        for j in range(batch_size):
            centers = []
            for i in range(num_spheres):
                # object center
                center = torch.mean(meshes_batch[j].verts_packed()[num_vehicle_vertices + i * num_sphere_vertices : num_vehicle_vertices + (i+1) * num_sphere_vertices], dim=0)
                centers += [center.repeat(num_sphere_vertices,1)]
            centers = torch.cat(centers,dim=0)
            
            # vehicle center
            center_v = torch.mean(meshes_batch[j].verts_packed()[:num_vehicle_vertices], dim=0)
            center_v = center_v.repeat(meshes_batch[j].verts_packed()[num_vehicle_vertices:].shape[0],1)
            
            deform_verts3 = torch.full(meshes_batch[0].verts_packed().shape, 0.0, device=self.device)

            offset_loc = sphere_location - centers + center_v

            deform_verts3[num_vehicle_vertices:] = offset_loc
            
            deform_verts3_l += [deform_verts3]
            
        deform_verts3_l = torch.cat(deform_verts3_l, dim=0)
        meshes_batch.offset_verts_(vert_offsets_packed=deform_verts3_l)
        
        color = self.soft_argmax(texture_map)
        color = color.double() @ input_color.double()
        mult = 320 // res
        
        for a in range(num_spheres):
            for i in range(res):
                for j in range(res):
                    meshes_batch.textures._maps_padded[:,mult * i:mult * (i+1), 512 + 320 * a + mult * j: 512 + 320 * a + mult * (j+1),:] = color[res * i + j]
        
        if adv_texture is not None:
            meshes_batch.textures._maps_padded[:,:512, :512,:] = adv_texture
        
        return meshes_batch
    
    def soft_argmax(self, core_textures, temp=0.02):
        """
        This function implements one-hot softmax approximation of argmax (using low temperature softmax).
        Exact formula:
        
        z_i = exp(ln(x_i) / temp) / [sum_j(exp(ln(x_j) / temp))]
        """
        numerator = torch.exp(torch.log(core_textures) / temp)
        denominator = torch.sum(numerator, dim=1)
        soft_argmax = numerator / denominator.reshape(numerator.shape[0],1)
        
        return soft_argmax
    
    def render_ontensor_continuous(self, batch_size, image_batch, M, M2, sphere_location, iter_counter, adv_texture=None):
        distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(batch_size)

        M = M[:batch_size]
        
        n_vehicles_list = [1]*batch_size
        meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        
        meshes_batch_list2 = [[M2[0]] for i in range(batch_size)]
        
        meshes, locations_batch, offsets, rotations = self.randomly_place_meshes_multi(
            meshes_batch_list, 
            scaling_factors,
            distances,
            elevations,
            azimuths,
            intensities,
        )
        
        meshes_clone = [mesh[0].clone() for mesh in meshes]
        
        annotations = self.construct_annotations_files(locations_batch)
        
        num_vehicle_verts = M[0].verts_padded().shape[1]
        num_sphere_verts = M2[0].verts_padded().shape[1]
        
        new_meshes_spheres = [[] for i in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(sphere_location.shape[0]):
                modified_sphere = meshes_batch_list2[i][0].clone()
                new_meshes_spheres[i] += [modified_sphere]

        meshes_joined = []
        for i in range(batch_size):
            meshes[i] += new_meshes_spheres[i]
            meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))

        ##### Modified afterwards! #####
    
        meshes_batch = self.join_meshes_and_replace_textures(meshes_joined, batch_size, sphere_location, num_vehicle_verts, num_sphere_verts, adv_texture)
        
        
#         tot_vec = []
#         for i in range(len(distances)):
#             distance = distances[i]
#             elevation = elevations[i]
#             azimuth = azimuths[i]
        
#             R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)

#             cameras = FoVOrthographicCameras(
#                 device=self.device,
#                 R=R,
#                 T=T,
#                 scale_xyz=scaling_factors
#             )
            
#             cam_center = cameras[0].get_camera_center()
            
#             R = R.to(self.device)
            
#             d_camZ = 1
            
#             location = torch.mean(meshes_batch[i].verts_packed()[num_vehicle_verts:], dim=0).reshape(1,3)
            
#             transformed_loc_x = d_camZ * R[0,0,2] + cam_center[0,0]
#             transformed_loc_y = d_camZ * R[0,1,2] + cam_center[0,1]
#             transformed_loc_z = d_camZ * R[0,2,2] + cam_center[0,2]
#             vec = torch.tensor([transformed_loc_x, transformed_loc_y, transformed_loc_z], device=self.device)
            
#             deform_verts3 = torch.full(meshes_batch[0].verts_packed().shape, 0.0, device=self.device)

#             vec = vec.repeat(num_sphere_verts,1)
#             deform_verts3[num_vehicle_verts:] = vec
#             tot_vec += [deform_verts3]
#         tot_vec = torch.cat(tot_vec)
#         meshes_batch.offset_verts_(vert_offsets_packed=tot_vec)
        
        synthetic_images = self.render_batch(
            meshes_batch, 
            image_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )
                
#         total_loss = 0
        
#         for i in range(sphere_location.shape[0]):
#             center_v = torch.mean(meshes_clone[i].verts_packed(),dim=0)
#             pointcloud = Pointclouds(points=[sphere_location[i].reshape(1,3) + center_v.reshape(1,3)])
#             total_loss += point_mesh_face_distance(meshes_clone[0], pointcloud)
        
#         if iter_counter % 100 == 0:
#             from pytorch3d.io import IO
#             IO().save_mesh(meshes_batch[0], f"meshes/{iter_counter}.obj", include_textures=True)
        

        return synthetic_images #, total_loss, synthetic_images_before_mod
    
#     def render_ontensor_camera_ray(self, batch_size, image_batch, M, M2, sphere_location, iter_counter, d_camZ, adv_texture=None):
#         distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(batch_size)

#         M = M[:batch_size]
        
#         n_vehicles_list = [1]*batch_size
#         meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        
#         meshes_batch_list2 = [[M2[0]] for i in range(batch_size)]
        
#         meshes, locations_batch, offsets, rotations = self.randomly_place_meshes_multi(
#             meshes_batch_list, 
#             scaling_factors,
#             distances,
#             elevations,
#             azimuths,
#             intensities,
#         )
        
#         meshes_clone = [mesh[0].clone() for mesh in meshes]
        
#         annotations = self.construct_annotations_files(locations_batch)
        
#         num_vehicle_verts = M[0].verts_padded().shape[1]
#         num_sphere_verts = M2[0].verts_padded().shape[1]
        
#         new_meshes_spheres = [[] for i in range(batch_size)]
        
#         for i in range(batch_size):
#             for j in range(sphere_location.shape[0]):
#                 modified_sphere = meshes_batch_list2[i][0].clone()
#                 new_meshes_spheres[i] += [modified_sphere]

#         meshes_joined = []
#         for i in range(batch_size):
#             meshes[i] += new_meshes_spheres[i]
#             meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))

#         ##### Modified afterwards! #####
    
#         meshes_batch = self.join_meshes_and_replace_textures(meshes_joined, batch_size, sphere_location, num_vehicle_verts, num_sphere_verts, adv_texture)
        
        
#         tot_vec = []
#         for i in range(len(distances)):
#             distance = distances[i]
#             elevation = elevations[i]
#             azimuth = azimuths[i]
        
#             R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)

#             cameras = FoVOrthographicCameras(
#                 device=self.device,
#                 R=R,
#                 T=T,
#                 scale_xyz=scaling_factors
#             )
            
#             cam_center = cameras[0].get_camera_center()
            
#             R = R.to(self.device)
            
            
#             location = torch.mean(meshes_batch[i].verts_packed()[num_vehicle_verts:], dim=0).reshape(1,3)
            
#             transformed_loc_x = d_camZ * R[0,0,2] + cam_center[0,0]
#             transformed_loc_y = d_camZ * R[0,1,2] + cam_center[0,1]
#             transformed_loc_z = d_camZ * R[0,2,2] + cam_center[0,2]
#             vec = torch.cat([transformed_loc_x, transformed_loc_y, transformed_loc_z],0)
            
#             deform_verts3 = torch.full(meshes_batch[0].verts_packed().shape, 0.0, device=self.device)

#             vec = vec.repeat(num_sphere_verts,1)
#             deform_verts3[num_vehicle_verts:] = vec
#             tot_vec += [deform_verts3]
#         tot_vec = torch.cat(tot_vec)
#         meshes_batch.offset_verts_(vert_offsets_packed=tot_vec)
        
#         synthetic_images = self.render_batch(
#             meshes_batch, 
#             image_batch, 
#             elevations, 
#             azimuths,
#             lights_directions,
#             scaling_factors=scaling_factors,
#             intensities=intensities,
#             distances=distances,
#             image_size=384
#         )
                
#         total_loss = 0
        
#         for i in range(sphere_location.shape[0]):
#             center_cube = torch.mean(meshes_batch[i].verts_packed()[num_vehicle_verts:],dim=0)
#             pointcloud = Pointclouds(points=[center_cube.reshape(1,3)])
#             total_loss += point_mesh_face_distance(meshes_clone[0], pointcloud)
        
# #         if iter_counter % 100 == 0:
# #             from pytorch3d.io import IO
# #             IO().save_mesh(meshes_batch[0], f"meshes/{iter_counter}.obj", include_textures=True)
        

#         return synthetic_images, total_loss
    
    
    def render_ontensor_optimizeCubeTexture(self, batch_size, image_batch, M, M2, texture_map, iter_counter, sphere_location, directory, adv_texture=None, input_color=None, res=16):
        distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(batch_size)

        M = M[:batch_size]
        
        n_vehicles_list = [1]*batch_size
        meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        
        meshes_batch_list2 = [[M2[0]] for i in range(batch_size)]
        
        meshes, locations_batch, offsets, rotations = self.randomly_place_meshes_multi(
            meshes_batch_list, 
            scaling_factors,
            distances,
            elevations,
            azimuths,
            intensities,
        )
        annotations = self.construct_annotations_files(locations_batch)
        
        num_vehicle_verts = M[0].verts_padded().shape[1]
        num_sphere_verts = M2[0].verts_padded().shape[1]
        
        new_meshes_spheres = [[] for i in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(sphere_location.shape[0]):
                modified_sphere = meshes_batch_list2[i][0].clone()
                new_meshes_spheres[i] += [modified_sphere]

        meshes_joined = []
        for i in range(batch_size):
            meshes[i] += new_meshes_spheres[i]
            meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))

        ##### Modified afterwards! #####
        
        meshes_batch = self.join_meshes_and_replace_cube_textures(meshes_joined, batch_size, sphere_location, texture_map, num_vehicle_verts, num_sphere_verts, adv_texture=adv_texture, input_color=input_color)
        
        if iter_counter % 100 == 0:
            from pytorch3d.io import IO
            IO().save_mesh(meshes_batch[0], f"meshes/{iter_counter}.obj", include_textures=True)
        
        
        texture_map_new = meshes_batch.textures._maps_padded[0, :320, 512:512 + 320]
        
        from torchvision.utils import save_image
        texture_map_export = texture_map_new.clone().detach().cpu().permute(2,0,1)
        save_image(texture_map_export,f'./{directory}/texture_map/iteration_{iter_counter:05d}.png')
        
        
        synthetic_images = self.render_batch(
            meshes_batch, 
            image_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )

        return synthetic_images
    
    def render_ontensor_limited_num(self, batch_size, image_batch, M, M2, silhouette_vector, sphere_color, sphere_location, iter_counter, input_color, vert_indices, shape):
        distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(batch_size)

        M = M[:batch_size]
        
        n_vehicles_list = [1]*batch_size
        meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        
        meshes_batch_list2 = [[M2[0]] for i in range(batch_size)]
        
        meshes, locations_batch, offsets, rotations = self.randomly_place_meshes_multi(
            meshes_batch_list, 
            scaling_factors,
            distances,
            elevations,
            azimuths,
            intensities,
        )
        annotations = self.construct_annotations_files(locations_batch)
        
        num_vehicle_verts = M[0].verts_padded().shape[1]
        num_sphere_verts = M2[0].verts_padded().shape[1]
        
        new_meshes_spheres = [[] for i in range(batch_size)]
       
        for i in range(batch_size):
            for j in range(len(vert_indices)):
                modified_sphere = meshes_batch_list2[i][0].clone()

                mean_x = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][0].item()
                mean_y = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][1].item()
                mean_z = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][2].item()

                rotation_vector = -meshes[i][0].verts_normals_packed()[vert_indices[j]]
                
                original = torch.tensor([0,1,0], device=self.device)
                
                mesh_rotation1 = self.rotation_matrix_from_vectors(original, rotation_vector)
                
                mesh_rotation = torch.matmul(mesh_rotation1, modified_sphere.verts_packed().data.T).T - modified_sphere.verts_packed()
                modified_sphere.offset_verts_(vert_offsets_packed=mesh_rotation)
                
                offset_loc = torch.tensor(meshes[i][0].verts_packed()[vert_indices[j]] - torch.tensor([mean_x,mean_y,mean_z], device=self.device),device=self.device)
                modified_sphere.offset_verts_(vert_offsets_packed=offset_loc)

                new_meshes_spheres[i] += [modified_sphere]
                
                

        meshes_joined = []
        for i in range(batch_size):
            meshes[i] += new_meshes_spheres[i]
            meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))
        # meshes_batch = join_meshes_as_batch(meshes_joined)

        ##### Modified afterwards! #####
        
        # meshes_batch = self.join_meshes_and_replace_textures(meshes_joined, batch_size, silhouette_vector, sphere_location, sphere_color, input_color, num_vehicle_verts, num_sphere_verts, vert_indices, shape)
        meshes_batch = self.join_meshes_and_replace_textures(meshes_joined, batch_size, silhouette_vector, sphere_location, sphere_color, input_color, num_vehicle_verts, num_sphere_verts, vert_indices, shape='cube')
        
        if iter_counter % 100 == 0:
            from pytorch3d.io import IO
            IO().save_mesh(meshes_batch[0], f"meshes/{iter_counter}.obj", include_textures=True)
        
        synthetic_images = self.render_batch(
            meshes_batch, 
            image_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )

        return synthetic_images
   
    def render_ontensor_validation(self, batch_size, image_batch, M, M2, sphere_location, iter_counter, adv_texture=None, input_color=None):
        distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(batch_size)

        M = M[:batch_size]
        n_vehicles_list = [1]*batch_size
        meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        
        meshes_batch_list2 = [[M2[0]] for i in range(batch_size)]

        
        meshes, locations_batch, offsets, rotations = self.randomly_place_meshes_multi(
            meshes_batch_list, 
            scaling_factors,
            distances,
            elevations,
            azimuths,
            intensities,
        )
        annotations = self.construct_annotations_files(locations_batch)

        meshes_def = []
        for i in range(batch_size):
            meshes_def.append(join_meshes_as_scene([meshes_batch_list[i][0]]).to(self.device))

        synthetic_images = self.render_batch(
            meshes_def, 
            image_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )
            
        num_vehicle_verts = M[0].verts_padded().shape[1]
        num_sphere_verts = M2[0].verts_padded().shape[1]
        
        new_meshes_spheres = [[] for i in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(sphere_location.shape[0]):
                modified_sphere = meshes_batch_list2[i][0].clone()
                new_meshes_spheres[i] += [modified_sphere]

        meshes_joined = []
        for i in range(batch_size):
            meshes[i] += new_meshes_spheres[i]
            meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))

        ##### Modified afterwards! #####
    
        meshes_batch = self.join_meshes_and_replace_textures(meshes_joined, batch_size, sphere_location, num_vehicle_verts, num_sphere_verts, adv_texture, input_color)
        
        img_counter = iter_counter

        if iter_counter % 100 == 0:
            from pytorch3d.io import IO
            IO().save_mesh(meshes_batch[0], f"meshes/{iter_counter}.obj", include_textures=True)
            
#         image = PIL.Image.open(f'meshes/{iter_counter}.png')
        
#         transform = transforms.Compose([
#             transforms.PILToTensor()
#         ])

#         # transform = transforms.PILToTensor()
#         # Convert the PIL image to Torch tensor
#         img_tensor = transform(image)

#         imported_texture = img_tensor.permute(1,2,0) / 255
        
#         imported_texture[:300, 512:,:] = texture_map
        
#         save_image(imported_texture.detach().cpu().permute(2,0,1), f"meshes/{iter_counter}.png", quality=100)
        
        
        synthetic_images_modified = self.render_batch(
            meshes_batch, 
            image_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )

        return synthetic_images, synthetic_images_modified, annotations
    
    def render_ontensor_validation1(self, batch_size, image_batch, M, M2, sphere_location, iter_counter, adv_texture=None, input_color=None):
        distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(batch_size)

        M = M[:batch_size]
        n_vehicles_list = [1]*batch_size
        meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        
        meshes_batch_list2 = [[M2[0]] for i in range(batch_size)]

        
        meshes, locations_batch, offsets, rotations = self.randomly_place_meshes_multi(
            meshes_batch_list, 
            scaling_factors,
            distances,
            elevations,
            azimuths,
            intensities,
        )
        annotations = self.construct_annotations_files(locations_batch)

        meshes_def = []
        for i in range(batch_size):
            meshes_def.append(join_meshes_as_scene([meshes_batch_list[i][0]]).to(self.device))

        synthetic_images = self.render_batch(
            meshes_def, 
            image_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )
            
#         num_vehicle_verts = M[0].verts_padded().shape[1]
#         num_sphere_verts = M2[0].verts_padded().shape[1]
        
#         new_meshes_spheres = [[] for i in range(batch_size)]
        
#         for i in range(batch_size):
#             for j in range(sphere_location.shape[0]):
#                 modified_sphere = meshes_batch_list2[i][0].clone()
#                 new_meshes_spheres[i] += [modified_sphere]

#         meshes_joined = []
#         for i in range(batch_size):
#             meshes[i] += new_meshes_spheres[i]
#             meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))


        ##### Modified afterwards! #####
    
        meshes_batch = join_meshes_as_batch(meshes_def)
        if adv_texture != None:
            meshes_batch.textures._maps_padded[:,:512,:512,:] = adv_texture[:,:,:3]
        
        img_counter = iter_counter

        # if iter_counter % 100 == 0:
        #     from pytorch3d.io import IO
        #     IO().save_mesh(meshes_batch[0], f"meshes/{iter_counter}.obj", include_textures=True)
            
#         image = PIL.Image.open(f'meshes/{iter_counter}.png')
        
#         transform = transforms.Compose([
#             transforms.PILToTensor()
#         ])

#         # transform = transforms.PILToTensor()
#         # Convert the PIL image to Torch tensor
#         img_tensor = transform(image)

#         imported_texture = img_tensor.permute(1,2,0) / 255
        
#         imported_texture[:300, 512:,:] = texture_map
        
#         save_image(imported_texture.detach().cpu().permute(2,0,1), f"meshes/{iter_counter}.png", quality=100)
        
        
        synthetic_images_modified = self.render_batch(
            meshes_batch, 
            image_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )

        return synthetic_images, synthetic_images_modified, annotations
    
    def render_ontensor_evaluateVertexAttackHeatMap(self, batch_size, image_batch, M, M2, iter_counter, shape):
        distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(batch_size)

        M = M[:batch_size]
        n_vehicles_list = [1]*batch_size
        meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        
        meshes_batch_list2 = [[M2[0]] for i in range(batch_size)]

        
        meshes, locations_batch, offsets, rotations = self.randomly_place_meshes_multi(
            meshes_batch_list, 
            scaling_factors,
            distances,
            elevations,
            azimuths,
            intensities,
        )
        annotations = self.construct_annotations_files(locations_batch)
        
        new_meshes_spheres = [[] for a in range(batch_size)]
        meshes_joined = []
        for i in range(batch_size):
#             modified_sphere = meshes_batch_list2[i][0].clone()

#             mean_x = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][0].item()
#             mean_y = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][1].item()
#             mean_z = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][2].item()

#             rotation_vector = -meshes[i][0].verts_normals_packed()[49]

#             original = torch.tensor([0,1,0], device=self.device)

#             mesh_rotation1 = self.rotation_matrix_from_vectors(original, rotation_vector)

#             mesh_rotation = torch.matmul(mesh_rotation1, modified_sphere.verts_packed().data.T).T - modified_sphere.verts_packed()
#             modified_sphere.offset_verts_(vert_offsets_packed=mesh_rotation)

#             offset_loc = torch.tensor(meshes[i][0].verts_packed()[49] - torch.tensor([mean_x,mean_y,mean_z], device=self.device),device=self.device)
#             modified_sphere.offset_verts_(vert_offsets_packed=offset_loc)

#             new_meshes_spheres[i] += [modified_sphere]

            # meshes_joined.append(join_meshes_as_scene(meshes[i]+new_meshes_spheres[i]).to(self.device))
            meshes_joined.append(join_meshes_as_scene(meshes_batch_list[i]).to(self.device))

        meshes_batch = join_meshes_as_batch(meshes_joined)

        synthetic_images_modified = self.render_batch(
                meshes_batch, 
                image_batch, 
                elevations, 
                azimuths,
                lights_directions,
                scaling_factors=scaling_factors,
                intensities=intensities,
                distances=distances,
                image_size=384
            )
        all_data = [synthetic_images_modified]
        
        num_vehicle_verts = M[0].verts_padded().shape[1]
        num_sphere_verts = M2[0].verts_padded().shape[1]
        
        for vertex_number in range(num_vehicle_verts):
            new_meshes_spheres = [[] for a in range(batch_size)]
            meshes_joined = []
            for i in range(batch_size):
                modified_sphere = meshes_batch_list2[i][0].clone()

                mean_x = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][0].item()
                mean_y = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][1].item()
                mean_z = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][2].item()

                rotation_vector = -meshes[i][0].verts_normals_packed()[49]

                original = torch.tensor([0,1,0], device=self.device)

                mesh_rotation1 = self.rotation_matrix_from_vectors(original, rotation_vector)

                mesh_rotation = torch.matmul(mesh_rotation1, modified_sphere.verts_packed().data.T).T - modified_sphere.verts_packed()
                modified_sphere.offset_verts_(vert_offsets_packed=mesh_rotation)

                offset_loc = torch.tensor(meshes[i][0].verts_packed()[49] - torch.tensor([mean_x,mean_y,mean_z], device=self.device),device=self.device)
                modified_sphere.offset_verts_(vert_offsets_packed=offset_loc)
                
                new_meshes_spheres[i] += [modified_sphere]
                
                # Attach another sphere!

                modified_sphere = meshes_batch_list2[i][0].clone()

                mean_x = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][0].item()
                mean_y = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][1].item()
                mean_z = torch.mean(meshes_batch_list2[i][0].verts_padded(), dim=1)[0][2].item()

                rotation_vector = -meshes[i][0].verts_normals_packed()[vertex_number - 1]

                original = torch.tensor([0,1,0], device=self.device)

                mesh_rotation1 = self.rotation_matrix_from_vectors(original, rotation_vector)

                mesh_rotation = torch.matmul(mesh_rotation1, modified_sphere.verts_packed().data.T).T - modified_sphere.verts_packed()
                modified_sphere.offset_verts_(vert_offsets_packed=mesh_rotation)

                offset_loc = torch.tensor(meshes[i][0].verts_packed()[vertex_number - 1] - torch.tensor([mean_x,mean_y,mean_z], device=self.device),device=self.device)
                modified_sphere.offset_verts_(vert_offsets_packed=offset_loc)

                new_meshes_spheres[i] += [modified_sphere]

                meshes_joined.append(join_meshes_as_scene(meshes[i]+new_meshes_spheres[i]).to(self.device))

            meshes_batch = join_meshes_as_batch(meshes_joined)

            synthetic_images_modified = self.render_batch(
                meshes_batch, 
                image_batch, 
                elevations, 
                azimuths,
                lights_directions,
                scaling_factors=scaling_factors,
                intensities=intensities,
                distances=distances,
                image_size=384
            )
            
            all_data += [synthetic_images_modified]

        return all_data, annotations

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        vec1 = vec1.to(torch.float)
        vec2 = vec2.to(torch.float)
        a, b = (vec1 / torch.norm(vec1)).reshape(3), (vec2 / torch.norm(vec2)).reshape(3)
        v = torch.cross(a, b)
        c = torch.dot(a, b)
        s = torch.linalg.norm(v)
        kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],device=self.device)
        rotation_matrix = torch.eye(3,device=self.device) + kmat + torch.matmul(kmat,kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    

if __name__ == '__main__':
    print("Differentiable renderer")
    dif = differentiablerenderer()
    dif.render()



