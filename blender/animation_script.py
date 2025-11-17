import bpy
import bpy_extras
import numpy as np
import math
import mathutils
import random
from typing import Tuple


#x rotation matrix given input in degrees
def rotate2DX(theta):
    theta = math.radians(theta)
    return np.array([[1,0,0],[0, math.cos(theta), -1 * math.sin(theta)],[0, math.sin(theta), math.cos(theta)]])
#y rotation matrix given input in degrees
def rotate2DY(theta):
    theta = math.radians(theta)
    return np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-1 * math.sin(theta), 0,  math.cos(theta)]])
#z rotation matrix given input in degrees
def rotate2DZ(theta):
    theta = math.radians(theta)
    return np.array([[math.cos(theta),-1*math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0, 0, 1]])
#xyz rotation matrix given x,y,z rotation in degrees
def rotate3D(xTheta, yTheta, zTheta):
    xRot = rotate2DX(xTheta)
    yRot = rotate2DY(yTheta)
    zRot = rotate2DZ(zTheta)
    return xRot @ yRot @ zRot


#the joint order for joint coordinates to be recorded in each frame
def setOrder() -> dict:
    order = dict() 
    order['head'] = None #1
    order['neck2'] = None #2
    order['torso'] = None #3
    order['upperleg.R'] = None #4
    order['frontleg.R'] = None #5
    order['forearm.R'] = None #6
    order['upperleg.L'] = None #7
    order['frontleg.L'] = None #8
    order['forearm.L'] = None #9
    order['pelvis'] = None #10
    order['pelvis.R'] = None #11
    order['shin.R'] = None #12
    order['backfoot.R'] = None #13
    order['pelvis.L'] = None #14
    order['shin.L'] = None #15
    order['backfoot.L'] = None #16
    order['tail'] = None #17
    order['lefteye'] = None #18
    order['righteye'] = None #19
    return order
        

class BoneWrapper:
    
    '''
    Wrapper class for a single blender bone that is part of an armature.
    Handles parent-aware rotation, frame-by-frame animation, and blender-space to image-space coordinate conversion
    
    Args:
        armature: bpy.data.objects[armature_name] the rigged mesh
        bone_name: the name of the bone as named in the armature
        init_rot: the x, y, and x initial angles of this bone in degrees
        all_vae_priors: an array of shape (`n_frames`, 3*`n_bones`), where the VAE has pre-computed the xyz angles 
                    for `n_bones` bones over `n_frames` frames.
        vae_prior_indices: the 3 indicies corresponding to the 3 columns in the VAE prior array for the xyz coordinates of this bone
                    across all `n_frames` frames
        correction_needed: does this bone need additional correction due to the specifc yaw, pitch, roll of this armature?
                    Only applied to bones that rely on VAE priors.
        x_rot_range: the lower and upper bounds of random rotation in degrees for this bone, across the X axis
                    Only applied to bones that do not rely on VAE priors
        y_rot_range: the lower and upper bounds of random rotation in degrees for this bone, across the Y axis
                    Only applied to bones that do not rely on VAE priors
        z_rot_range: the lower and upper bounds of random rotation in degrees for this bone, across the Z axis
                    Only applied to bones that do not rely on VAE priors
        x_rot_func: some bones rely on the pre-computed X rotation for the Z rotation (only used if `z_rot_range` is `None`)
                        
    '''
    
    def __init__(self, armature, bone_name: str = None, init_rot: Tuple[int, int, int] = (0,0,0), all_vae_priors: np.ndarray = None, vae_prior_indices: Tuple[int, int, int] = None, correction_needed: bool = False, 
    x_rot_range: Tuple[int, int] = (0,0), y_rot_range: Tuple[int, int] = (0,0), z_rot_range: Tuple[int, int] = None, z_rot_func = None):
        self.armature = armature
        assert bone_name in self.armature.pose.bones
        self.bone_name = bone_name
        self.poseBone = self.armature.pose.bones[bone_name]
        #self.poseBone.matrix_basis = np.eye(4)
        self.init_rot = init_rot
        
        #record the parent of this bone
        self.parent = self.poseBone.parent
        
        #if this bone does not rely on VAE priors
        self.vae_priors = None
        self.max_frames = 100
        
        #if this bone relies on VAE priors
        if vae_prior_indices is not None and all_vae_priors is not None:
            
            #only record the 3 columns from the VAE prior matrix that correspond to the XYZ angles of this bone per frame
            self.vae_priors = all_vae_priors[:, vae_prior_indices]
            
            #this bone cannot be animated beyond the number of frames given by the VAE
            self.max_frames = self.vae_priors.shape[0]
            
            
        self.correction_needed = correction_needed
        self.x_rot_range = x_rot_range
        self.y_rot_range = y_rot_range
        self.z_rot_range = z_rot_range
        self.z_rot_func = z_rot_func
        
        
    def get_rand_rot(self) -> Tuple[int, int, int]:
        
        '''
        Generates random X, Y, and Z angle rotation for this bone, under the given ranges/functions
        
        Args: None
        
        Returns: X, Y, Z rotation in degrees
        '''
        
        #generate X rotation
        xnoise = random.randint(self.x_rot_range[0], self.x_rot_range[1])
        
        #generate Y rotation 
        ynoise = random.randint(self.y_rot_range[0], self.y_rot_range[1])
        
        #if the Z rotation depends on the X rotation
        if not self.z_rot_range:
            znoise = self.z_rot_func(xnoise)
            
        #if the Z rotation is also randomized
        else:
            znoise = random.randint(self.z_rot_range[0], self.z_rot_range[1])
        return (xnoise, ynoise, znoise)
        
    def rotate(self, frame_number: int) -> None:
        
        '''
        Rotates this bone with respect to its parent in the given frame, adds a keyfame to the animation for this bone
        
        Args:
            frame_number: the index of the keyframe
            
        Returns: None
        '''
    
        
        if self.parent is None:
            return
            
        #assume no 3D rotation for this bone (3x3 identity)
        poseRot3D = np.eye(3)
            
        #this bone uses VAE priors
        if self.vae_priors is not None:
            
            #cancel rotation if VAEdoes not have a prior for the given frame
            if frame_number > self.max_frames:
                return
            
            #get the x, y, z rotation angles in degrees, for the given frame, according to the VAE
            xyz_prior = self.vae_priors[frame_number, :]
            
            #account for offset from intitial X rotation of this bone
            diffX = xyz_prior[0] - self.init_rot[0] 
            
            #double account for offset if more correction is needed
            #and then compute the 3x3 rotation matrix based on the x, y, z angles in degrees
            if self.correction_needed:
                poseRot3D = rotate3D(2*diffX, 0, 0)
                
            else:
                poseRot3D = rotate3D(diffX, 0, 0)
                
        
                
        
        #this bone does not use VAE priors, so generate rotation randomly         
        else:
                
            #use the ranges/functions for this bone to generate the random x, y, z angles in degrees
            #and then compute the 3x3 rotation matrix
            noiseX, noiseY, noiseZ = self.get_rand_rot()
            poseRot3D = rotate3D(noiseX,noiseY,noiseZ)
            
        
        
    
        
        #retrieve 3x3 rotation matrix of the parent of this bone
        parentPoseRot3D = np.array(self.parent.matrix_basis)[:3, :3]
        
        
        
        
    
        
        #apply the parent rotation to this bone's rotation using inverse kinematics
        finalPoseRot3D = np.linalg.inv(parentPoseRot3D) @ poseRot3D
        
        
        #apply the parent-aware 3D rotation matrix of this bone using the blender API
        #which requires a 4D rotation matrix (quaternion)
        poseRot4D = np.eye(4)
        poseRot4D[:3,:3] = finalPoseRot3D 
            
        
            
        self.poseBone.matrix_basis = poseRot4D.T
        
        
        #update the given frame of the animation with this bone's rotation
        blSet = self.poseBone.keyframe_insert('rotation_quaternion', frame = frame_number)
        
    
    def get_joint_coordinates(self, scene, camera, height, width, head: bool = False) -> Tuple[int, int]:
        
        '''
        Converts the head or tail coordinates (3D Blender Space) of this bone to 2D cartesian coordinates in the (`width` x `height`) rendered image
        using the lens of the given scene and camera
        
        Args:
            scene: the scene object in Blender(bpy.context.scene)
            camera: the camera object in Blender (bpy.data.objects[camera_name])
            height: the height of the rendered image in pixels
            width: the width of the rendered image in pixels
            head: use the head or tail of this bone as the joint?
            
        Returns: (x, y) coordinates of this joint in the rendered image
        '''
        
        #get the 3D Blender space coordinates of the head of this bone
        if head:
            world_location_3D = armature.matrix_world @ self.poseBone.head
        #get the 3D Blender space coordinates of the tail of this bone
        else:
            world_location_3D = armature.matrix_world @ self.poseBone.tail
          
        #get the 2D coordinates of this joint according to the lens of the given scene and camera
        #which are scaled between 0 and 1
        cco = bpy_extras.object_utils.world_to_camera_view(scene, camera, world_location_3D)
        
        #the true X coordinate in the image (width is X-axis in cartesian space)
        x=round(cco.x * width)
        
        #the true Y coordinate in the image (height is Y-axis in cartesian space, 
        #but the cco origin is top left corner and need to flip so origin is bottom left corner)
        y=height-round(cco.y * height)
        
        return (x, y)
        
               
        
            
if __name__ == '__main__':
    
    
    #set up scene
    scene = bpy.context.scene
    
    #image dimensions (currently set to 1920x1080)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale))
    width = render_size[0]
    height = render_size[1]
    
    print(width, height)
    
    #set up camera
    camera = bpy.data.objects["Camera"]
    
    #set up armature
    armature = bpy.data.objects["metarig"]
    
    #load full VAE priors (original)
    vae_prior_pth: str = "vae_prior.npz"
    vae_data_raw = np.load(vae_prior_pth)
    all_vae_priors = np.array(vae_data_raw['poses'])
    
    #use custom VAE priors
    all_vae_priors = np.load("my_vae_poses.npy")

    
    #set up bone wrappers in order
    bone_dict = dict()
    
    bone_dict['neck2'] = BoneWrapper(armature = armature, bone_name="neck2",
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(-10,10), y_rot_range = (0, 0), z_rot_range = (0, 0))
    
    bone_dict['torso'] = BoneWrapper(armature = armature, bone_name="torso", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(0,0), y_rot_range = (0, 0), z_rot_range = (0, 0))
    
    bone_dict['shoulder.R'] = BoneWrapper(armature = armature, bone_name="shoulder.R", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(0,0), y_rot_range = (0, 0), z_rot_range = (0,0))
    
    bone_dict['upperleg.R'] = BoneWrapper(armature = armature, bone_name="upperleg.R", 
    init_rot=(74,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(9,10,11))
    bone_dict['frontleg.R'] = BoneWrapper(armature = armature, bone_name="frontleg.R", 
    init_rot=(-50,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(12,13,14))
    bone_dict['forearm.R'] = BoneWrapper(armature = armature, bone_name="forearm.R", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(15,16,17))
    
    bone_dict['shoulder.L'] = BoneWrapper(armature = armature, bone_name="shoulder.L", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(0,0), y_rot_range = (0, 0), z_rot_range=(0,0))
    
    bone_dict['upperleg.L'] = BoneWrapper(armature = armature, bone_name="upperleg.L", 
    init_rot=(74,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(0,1,2))
    
    bone_dict['frontleg.L'] = BoneWrapper(armature = armature, bone_name="frontleg.L", 
    init_rot=(-50,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(3,4,5))
    
    bone_dict['forearm.L'] = BoneWrapper(armature = armature, bone_name="forearm.L", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(6,7,8))
    
    bone_dict['pelvis.R'] = BoneWrapper(armature = armature, bone_name="pelvis.R", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(0,0), y_rot_range = (0, 0), z_rot_range = (0,0))
    
    bone_dict['thigh.R'] = BoneWrapper(armature = armature, bone_name="thigh.R", 
    init_rot=(-90,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(27,28,29), correction_needed=True)
    
    bone_dict['shin.R'] = BoneWrapper(armature = armature, bone_name="shin.R", 
    init_rot=(31,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(30,31,32))
    
    bone_dict['backfoot.R'] = BoneWrapper(armature = armature, bone_name="backfoot.R", 
    init_rot=(-31,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(33,34,35))
    
    bone_dict['pelvis.L'] = BoneWrapper(armature = armature, bone_name="pelvis.L", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(0,0), y_rot_range = (0, 0), z_rot_range=(0,0))
    
    bone_dict['thigh.L'] = BoneWrapper(armature = armature, bone_name="thigh.L", 
    init_rot=(-90,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(18,19,20), correction_needed = True)
    
    bone_dict['shin.L'] = BoneWrapper(armature = armature, bone_name="shin.L", 
    init_rot=(31,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(21,22,23))
    
    bone_dict['backfoot.L'] = BoneWrapper(armature = armature, bone_name="backfoot.L", 
    init_rot=(-31,0,0), all_vae_priors=all_vae_priors, vae_prior_indices=(24,25,26))
    
    bone_dict['tail'] = BoneWrapper(armature = armature, bone_name="tail", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(-30,30), y_rot_range = (0, 0), z_rot_range = (0, 10))
    
    bone_dict['neck'] = BoneWrapper(armature = armature, bone_name="neck", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(-10,10), y_rot_range = (-10,10), z_rot_range = (-10, 10))
    
    bone_dict['head'] = BoneWrapper(armature = armature, bone_name="head", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(-30,30), y_rot_range = (-20,30), z_rot_range = (-40, 40))
    
    bone_dict['lefteye'] = BoneWrapper(armature = armature, bone_name="lefteye", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(0,0), y_rot_range = (0, 0), z_rot_func = lambda x: -x)
    
    bone_dict['righteye'] = BoneWrapper(armature = armature, bone_name="righteye", 
    init_rot=(0,0,0), all_vae_priors=all_vae_priors, x_rot_range=(0,0), y_rot_range = (0, 0), z_rot_func = lambda x: -x)
    
    
    #set number of frames in the animation
    numFrames = 1000
    
    #for each frame, store a list of (x,y) coordinates for each joint
    coordinates_2D_across_frames = []
    
    #for each frame in the animation
    for i in range(0, numFrames):
        
        #reset the order of joints to be recorded
        order = setOrder()
        
        #setup the blender scene with this frame
        scene.frame_set(i)
        
        #track each joint coordinate for this frame
        coordinates2D_frame_i = []
        
        #iterate over each bone in the armature, and its wrapper
        for bone_name, bone in bone_dict.items():
            
            
            #wrapper method for rotating this bone (parent rotation factored in) in the given scene and frame 
            bone.rotate(i)
   
            
            #wrapper method for retrieving the image cartesian coordinates of the joint (tail of this bone)
            #after the rotation had been aplied
            x, y = bone.get_joint_coordinates(scene, camera, height, width, False)
            
            #record the image cartesian coordinates of the joint 
            if bone_name in order:
                order[bone_name] = [x,y]
                
                #special case: for the purposes of h5, the pelvis is located at the head of the torso
                if bone_name == 'torso':
                    order['pelvis'] = list(bone.get_joint_coordinates(scene, camera, height, width, True))
                    
        for joint in order:
            coordinates2D_frame_i += order[joint]
        coordinates_2D_across_frames.append(coordinates2D_frame_i)
            
    np.save("blender_gt.npy", coordinates_2D_across_frames)
 

