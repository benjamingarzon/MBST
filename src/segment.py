"""
Functions for the MidBrain Segmentation Tool.
"""
# Author: Benjamin Garzon <benjamin.garzon@gmail.com>
# License: BSD 3 clause
from __future__ import division
from subprocess import call
from nipy import load_image, save_image
from nipy.core.api import Image, ImageList
import os
import numpy as np
import pickle
from collections import defaultdict
import warnings
import nibabel as nib
from nilearn.image import smooth_img

import sys

TOL = 1e-4
MAX_ITERS_EM = 200
MIN_ITERS_EM = 10
MAX_ITERS_ICM = 50
MAX_VOXELS_ICM = 1

TWOPI = 2*np.pi
EPS = 1e-10


SUBJECT_PREFIX = 'MBST_'

def write_image(x, coordmap, fname):
    """
    Write out an image.
        
    Parameters
    ----------

    x : numpy ndarray
        Array containing image intensities.

	coordmap : coordmap

    fname : string
        File name.
    
    """
    auxImg = Image(x.astype(np.float32), coordmap)

    warnings.simplefilter('ignore', FutureWarning)
    newimg = save_image(auxImg, fname)   
    warnings.simplefilter('default', FutureWarning)
    
def get_neighbours(x):
    """
    Get the neighbours of a point given the coordinates (6 points).
        
    Parameters
    ----------

    x : tuple
        3d coordinates of the center point.

    fname : string
        File name.
        
    Returns
    ----------
    
    neighbours: list
        List of 3d coordinates of the neighbours.
    
    """
    r = [-1, 0, 1]
    neighbours = [(x[0]+i,x[1]+j,x[2]+k) for i in r for j in r for k in r 
        if (i,j,k)!=(0,0,0)
        and abs(i)+abs(j)+abs(k) < 2]
    return(neighbours)         
      


def get_neighbours_2(x):
    """
    Get the neighbours of a point given the coordinates (27 points).
        
    Parameters
    ----------

    x : tuple
        3d coordinates of the center point.

    fname : string
        File name.
        
    Returns
    ----------
    
    neighbours: list
        List of 3d coordinates of the neighbours.
    
    """
    r = [-1, 0, 1]
    neighbours = [(x[0]+i,x[1]+j,x[2]+k) for i in r for j in r for k in r 
        if abs(i)+abs(j)+abs(k) < 2]
#    neighbours = [x]

    return(neighbours)    
            
def create_atlas(atlas_file, subjects_dir, sub_dir, subjects, structural_file, 
    structural_brain_file, structural_mask_file, parametric_file, mask_file, 
    label_file):
    """
    Create text file with a list of all files necessary to specify an atlas.
    
    Parameters
    ----------
    atlas_file : string
        File name for the text file with the atlas. 

    subjects_dir : string 
        Directory with the subjects. 
    
    sub_dir : string
        Name of the subdirectory with the images within the subject directory. 
    
    subjects : list 
        List of subject names. 
    
    structural_file : string
        File name for the structural images. 
    
    structural_brain_file : string 
        File name for brain extracted structural image. 
    
    structural_mask_file : string
        File name for the structural brain masks. 
    
    parametric_file : string 
        File name for the parametric images. 
    
    mask_file :  string
        File name for the masks. 
    
    label_file : string
        File name for the label images. 

    """
    
    print(("Creating atlas with subjects: %s")%subjects)
    f = open(atlas_file, 'w')

    for code, subject in enumerate(subjects):

        structural = subjects_dir + subject + sub_dir + structural_file
        structural_brain = subjects_dir + subject + sub_dir + \
            structural_brain_file
        structural_mask = subjects_dir + subject + sub_dir + \
            structural_mask_file
        parametric = subjects_dir + subject + sub_dir + parametric_file  
        mask = subjects_dir + subject + sub_dir + mask_file   
        label = subjects_dir + subject + sub_dir + label_file
    
        subject_code = "%s%03d"%(SUBJECT_PREFIX, code)
    
        line = ('%s:%s:%s:%s:%s:%s:%s\n')%(subject_code, structural, 
        structural_brain, structural_mask, parametric, mask, label)
        f.write(line)
    
    f.close()

def register(target_structural_file, target_structural_brain_file, 
    target_parametric_file, atlas_file, output_dir, clean_up, 
    overwrite):

    """
    Register the midbrain of all the instances of a given atlas to a target 
    subject.
    It uses FSL's registration tools http://fsl.fmrib.ox.ac.uk/fsl
        
    For each instance (subject) in the atlas, the registration is done in 3 
    stages:
        1 - Register non-linearly the structural images to localize the midbrain
        in the target image and apply the warps to the parametric image.
        2 - Crop the parametric image (QSM, ...) around the midbrain.
        3 - Register non-linearly the cropped parametric images to get an 
        optimal registration within the midbrain
        4 - Apply the registrations to the given label images.
    
        The structural and parametric file are assumed to be already aligned.
        
    Parameters
    ----------

    target_structural_file : string
        File name for the target structural image. 
    
    target_structural_brain_file : string 
        File name for the target brain extracted structural image. 
    
    target_parametric_file : string 
        File name for the target parametric image. 
    
    atlas_file : string 
        Text file with the list of files of the atlas (see generate_atlas). 
    
    output_dir : str
        Directory to save images generated in the registration.
    
    cleanup : boolean
        True - clean up unessential files after finishing to save space. 

    overwrite : boolean
        True - overwrite target_label_file if it already exists.
    """
    FSL_DIR_BIN = os.environ['FSL_DIR'] + '/bin/'

    # Create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read atlas
    f = open(atlas_file, 'r')
  
    
    for line in f:
        subject, structural_file, structural_brain_file, \
            structural_mask_file, parametric_file, mask_file, \
            label_file = line.replace('\n', '').split(':')
        
        # File names for the files that will be generated
        xfm_file = '%s/%s-xfm.mat'%(output_dir, subject)
        warp_file = '%s/%s-warp.nii.gz'%(output_dir, subject)
        fine_warp_file = '%s/%s-fine_warp.nii.gz'%(output_dir, subject)
        warped_parametric_file = '%s/%s-warped_param.nii.gz'%(output_dir, \
	    subject)
        cropped_parametric_file = '%s/%s-cr_param.nii.gz'%(output_dir, subject)
        cropped_target_parametric_file = \
            '%s/%s-cr_target_param.nii.gz'%(output_dir, subject)
        cropped_mask_file = '%s/%s-cr_mask.nii.gz'%(output_dir, subject)
        cropped_label_file = '%s/%s-cr_label.nii.gz'%(output_dir, subject)
        cropped_warped_label_file = '%s/%s-cr_warped_label.nii.gz'%(output_dir, 
            subject)
        cropped_warped_parametric_file = \
            '%s/%s-cr_warped_param.nii.gz'%(output_dir, subject)    
        invwarp_file = '%s/%s-invwarp.nii.gz'%(output_dir, subject)
        warped_mask_file = '%s/%s-warped_mask.nii.gz'%(output_dir, subject)
        
        # File with warped labels from this atlas subject
        warped_label_file = '%s/%s-warped_label.nii.gz'%(output_dir, subject)
        
        if (os.path.exists(warped_label_file) and overwrite) or \
            not os.path.exists(warped_label_file):
      
            print(('Registering subject %s.')%(subject))

            # Get mask dimensions        
            mask_image = load_image(mask_file)
            mask = mask_image.get_data()
            maxX = np.max(np.max(mask, axis=1), axis=1)
            firstX = np.where(maxX>0)[0][0]
            lastX = np.where(maxX>0)[0][-1] 
    
            maxY = np.max(np.max(mask, axis=0), axis=1)
            firstY= np.where(maxY>0)[0][0]
            lastY = np.where(maxY>0)[0][-1]
        
            maxZ = np.max(np.max(mask, axis=0), axis=0)
            firstZ = np.where(maxZ>0)[0][0]
            lastZ = np.where(maxZ>0)[0][-1]

            # Registration: target structural to structural
            command=[FSL_DIR_BIN+'flirt', '-in', target_structural_brain_file, 
                '-ref', structural_brain_file, '-omat', xfm_file]
            call(command)

            command=[FSL_DIR_BIN+'fnirt','--in=%s'%target_structural_file, 
                '--ref=%s'%structural_file, '--refmask=%s'%structural_mask_file,
                '--aff=%s'%xfm_file, '--cout=%s'%warp_file]
            call(command)

            command=[FSL_DIR_BIN+'applywarp','--in=%s'%target_parametric_file,
                '--ref=%s'%parametric_file,'--warp=%s'%warp_file,
                '--out=%s'%warped_parametric_file]
            call(command)

            # Invert warp
            command=[FSL_DIR_BIN+'invwarp','--ref=%s'%target_structural_file,
                '--warp=%s'%warp_file, '--out=%s'%invwarp_file] 
            call(command)

            print('Finished registering the structural image.')

            # Load parametric files
            parametric = load_image(parametric_file).get_data()
            target_parametric = load_image(warped_parametric_file).get_data()

    	    # Crop the arrays around mask
            cropped_parametric = parametric[firstX:lastX + 1, 
                firstY:lastY + 1, firstZ:lastZ + 1]
            cropped_target_parametric = target_parametric[firstX:lastX + 1, 
                firstY:lastY + 1, firstZ:lastZ + 1]
            cropped_mask = mask[firstX:lastX + 1, 
                firstY:lastY + 1, firstZ:lastZ + 1]
                
            write_image(cropped_parametric, mask_image.coordmap, 
                cropped_parametric_file)
            write_image(cropped_target_parametric, mask_image.coordmap, 
                cropped_target_parametric_file)
            write_image(cropped_mask, mask_image.coordmap, 
                cropped_mask_file)
        
            # Do finer registration
            command=[FSL_DIR_BIN+'fnirt','--in=%s'%cropped_parametric_file,
                '--inmask=%s'%cropped_mask_file,
                '--ref=%s'%cropped_target_parametric_file,
                '--refmask=%s'%cropped_mask_file,
                '--cout=%s'%fine_warp_file,
                '--intmod=global_linear',
                '--jacrange=.98,1.02']
            call(command)
            
            # Crop labels and apply registration
            label_image = load_image(label_file)
            labels = label_image.get_data()
        
            cropped_labels = labels[firstX:lastX + 1, 
                firstY:lastY + 1, firstZ:lastZ + 1]
            write_image(cropped_labels, label_image.coordmap, \
                cropped_label_file)

            command=[FSL_DIR_BIN+'applywarp','--in=%s'%cropped_label_file,
                '--ref=%s'%cropped_mask_file,
                '--warp=%s'%fine_warp_file,
                '--out=%s'%cropped_warped_label_file,
                '--interp=nn']
            call(command)
        
            # Apply to parametric map for the voting later
            command=[FSL_DIR_BIN+'applywarp','--in=%s'%cropped_parametric_file,
                '--ref=%s'%cropped_mask_file,
                '--warp=%s'%fine_warp_file,
                '--out=%s'%cropped_warped_parametric_file]
            call(command)
        
            # Put labels back into original space
            cropped_warped_labels = \
                load_image(cropped_warped_label_file).get_data()
    
            labels = np.zeros(labels.shape)
            labels[firstX:lastX + 1, firstY:lastY + 1, firstZ:lastZ + 1] = \
                cropped_warped_labels
    
            # Save labels
            write_image(labels, label_image.coordmap, warped_label_file)        

            # Apply inverse warp
            command=[FSL_DIR_BIN+'applywarp','--in=%s'%warped_label_file,
                '--ref=%s'%target_structural_file,
                '--warp=%s'%invwarp_file,
                '--out=%s'%warped_label_file,'--interp=nn']
            call(command)
        
            # Apply warp to mask
            command=[FSL_DIR_BIN+'applywarp','--in=%s'%mask_file,
                '--ref=%s'%target_structural_file,
                '--warp=%s'%invwarp_file,
                '--out=%s'%warped_mask_file,'--interp=nn']
            call(command)        
    
            # Done with all the warps   
            print('Finished applying the warps.')
            
        else:
            print('Registered labels already exist for subject %s.')%(subject)    
            
        # Clean up 
        if clean_up:
            for file_name in [
                xfm_file, 
                warp_file,
                fine_warp_file,
                warped_parametric_file,
                cropped_parametric_file, 
                cropped_label_file, 
                cropped_warped_label_file, 
                invwarp_file
                ]:
                    try:
                        os.remove(file_name)
                    except OSError:
                        pass                
    
def fuse_labels(subject_list, output_dir, fused_file, average_mask_file, 
    votes_file):
    """
    Fuses the labels of all the registered instances in the atlas. 
    For each of the instances, correlation between the warped and target 
    intensities is computed to give larger weight to instances with better
    registration. A mask is also created fusing those of all the instances 
    in the atlas.
    
    Parameters
    ----------

    subject_list : list
        List of atlas instances.
    
    output_dir : string 
        Directory containing stored images and to output results. 
    
    fused_file : string 
        File name for calculated fused labels (to be used as priors).
    
    average_mask_file : string 
        File name for the average mask, obtained fusing all the masks.

    votes_file : string 
        Text file where the votes will be saved.
        
    """

    n_subjects = 0
    total_votes = 0
#    votes_f = open(output_dir + '/' + 'votes.txt', 'w')
    votes_f = open(votes_file, 'w')
    
    print('Fusing all registered labels to obtain priors.')
    for subject in subject_list:
    
        # File names for the files that will be used
        warped_label_file = '%s/%s-warped_label.nii.gz'%(output_dir, subject)
        cropped_mask_file = '%s/%s-cr_mask.nii.gz'%(output_dir, subject)
        cropped_warped_parametric_file = \
            '%s/%s-cr_warped_param.nii.gz'%(output_dir, subject)
        cropped_target_parametric_file = \
            '%s/%s-cr_target_param.nii.gz'%(output_dir, subject)

        warped_mask_file = '%s/%s-warped_mask.nii.gz'%(output_dir, subject)      
                  
        # Load images
        label_image = load_image(warped_label_file)
        warped_labels = label_image.get_data()

        warped_mask_image = load_image(warped_mask_file)
        warped_mask = warped_mask_image.get_data()

        cropped_mask = load_image(cropped_mask_file).get_data()
        cropped_warped_parametric = \
            load_image(cropped_warped_parametric_file).get_data()
        cropped_target_parametric = \
            load_image(cropped_target_parametric_file).get_data()
        
        # Compute correlation between warped and target intensities
        corr = np.corrcoef(cropped_warped_parametric[cropped_mask>0], 
            cropped_target_parametric[cropped_mask>0])[0,1]

        # Compute votes and save        
        if corr < 0: corr = 0
        vote = corr**2
        votes_f.write('%s: %f\n'%(subject, vote))
        
        # Create / update fused labels and mask
        if n_subjects == 0:
            fused_labels = warped_labels*vote
            fused_mask = warped_mask*vote
        else:
            fused_labels += warped_labels*vote
            fused_mask += warped_mask*vote
        
        total_votes += vote
        n_subjects += 1        

    # Compute votes
    fused_labels = (fused_labels/total_votes)
    fused_mask = (fused_mask/total_votes) > .5
    votes_f.close()
    
    # Write out resulting images
    write_image(fused_labels, label_image.coordmap, fused_file)    
    write_image(fused_mask, warped_mask_image.coordmap, average_mask_file)   

def smooth_map(input_file, output_file, fwhm):
    """
    Returns value of a Gaussian function.
    
    Parameters
    ----------

    input_file : string
        Input file name.
    
    output_file : string
        Output file name.

    fwhm : float
	FWHM
                 
    """
    output_image = smooth_img(input_file, fwhm=fwhm) 
    nib.save(output_image, output_file)

def norm(x, sigma2):
    """
    Returns value of a Gaussian function.
    
    Parameters
    ----------

    x : numpy ndarray
        x values.
    
    sigma2 : float
        Variance.

    Returns
    ----------

    y : numpy ndarray
        Value of the function evaluated at x.                    
    """
    y = np.exp(-0.5*x**2/sigma2) / np.sqrt(TWOPI*sigma2) + EPS 
    return(y)
           
def expectation_maximization(parametric_matrix, priors_matrix):
    """
    Estimate posteriors by expectation maximization.
    
    Parameters
    ----------
    parametric_matrix : numpy ndarray
        Matrix with intensities repeated for each class (n_voxels x n_classes).
    
    priors_matrix : numpy ndarray
        Matrix with priors for each class (n_voxels x n_classes).
         
         
    Returns
    ----------     
    pis : numpy ndarray
        Matrix with posteriors for each class (n_voxels x n_classes).          
        
    """

    n_classes = parametric_matrix.shape[1]
    n_voxels = parametric_matrix.shape[0]
    
    iteration = 1
    ratio = 1
    E = -1
    
    priors_matrix = np.hstack((priors_matrix, priors_matrix, priors_matrix))/3
    parametric_matrix = np.hstack((parametric_matrix, \
	parametric_matrix, parametric_matrix))
    pis = priors_matrix

    # Iterate ensuring a min of iterations to avoid local min
    while ((ratio > TOL) or (iteration < MIN_ITERS_EM)) \
        and (iteration < MAX_ITERS_EM):

        # Update parameter estimates
        means = np.sum(pis * parametric_matrix, axis=0, keepdims=True)/ \
            np.sum(pis, axis=0, keepdims=True)
        
        means_matrix = np.repeat(means, n_voxels, axis=0)

        sigma2 = np.sum(pis * \
            (parametric_matrix - means_matrix)**2, axis=0, keepdims=True)/ \
            np.sum(pis, axis=0, keepdims=True)
        
        sigma2_matrix = np.repeat(sigma2, n_voxels, 0)
     
        if iteration == 1:
        
            # Separate background classes
            means_matrix[:, :n_classes] = \
                means_matrix[:, :n_classes] \
                - 2*np.sqrt(sigma2_matrix[:, :n_classes])

            means_matrix[:, n_classes:2*n_classes] = \
                means_matrix[:, n_classes:2*n_classes] \
                + 2*np.sqrt(sigma2_matrix[:, n_classes:2*n_classes])

            sigma2_matrix = sigma2_matrix/9
        
        f = norm((parametric_matrix - means_matrix), sigma2_matrix) \
            * priors_matrix
       
        pis = f / np.repeat(np.sum(f, axis=1, keepdims=True), n_classes*3, 1)
        
        # Update likelihood
        E_old = E
        
        E = - np.sum(np.log(np.sum(f, axis=1)))
        ratio = (E - E_old)/E_old

        iteration += 1

    print('Expectation-maximization: Iterations: %d, Cost: %f.'%(iteration-1, 
        E))

    pis = pis[:, :n_classes] + pis[:, n_classes:2*n_classes] + \
        pis[:, 2*n_classes:]
    return(pis)

def estimate_spatial_relations(subject_list, output_dir, spatial_rel_file, 
    priors_file):
    """
    Estimates the probabilities of classes of adjacent voxels from the
    label images warped to the target subject.
    
    Parameters
    ----------

    subject_list : list
        List of atlas instances.
    
    output_dir : string 
        Directory containing stored images and to output results. 
    
    spatial_rel_file : string 
        File name for the pickle file to store the spatial relations as a dict.
    The dict is indexed as: 
    (center coords, neighbour coords, center class, neighbour class):probability 
    
    priors_file : string 
        File name for the priors.
        
    """
    
    # Load priors
    priors_image = load_image(priors_file)
    priors = priors_image.get_data()

    mask = np.sum(priors, 3) > 0
    n_voxels = np.sum(mask > 0)
    print ("Number of voxels to estimate spatial relations for: %d.\n")%n_voxels
    
    spatial_rel =  defaultdict(float) 
        
    n_subjects = len(subject_list)

    coords = np.where(mask)
    points = set(zip(coords[0], coords[1], coords[2]))
        
    for subject in subject_list:
        print ("Estimating spatial relations for subject %s.\n")%subject
        
        # Load data
        warped_label_file = '%s/%s-warped_label.nii.gz'%(output_dir, subject)
        label_image = load_image(warped_label_file)
        warped_labels = label_image.get_data().astype(np.int8)
        shape = warped_labels.shape
       
        n_classes = shape[3]
        segmentation = np.zeros(shape[:3], dtype=np.int8)
        
        # Create segmentation image
        for i in range(n_classes):
            segmentation = segmentation + (i + 1) * warped_labels[:, :, :, i] 
        
        # Estimate spatial relations    
        for point in points:
            neighbours = get_neighbours(point)
            
            for neighbour in neighbours:
                index = (point, neighbour,
                    segmentation[point[0], point[1], point[2]], 
                    segmentation[neighbour[0], neighbour[1], neighbour[2]])
                spatial_rel[index] += 1./n_subjects
    
    # Smooth spatial relations
    spatial_rel_smoothed =  defaultdict(float) 
    spatial_rel_normalization =  defaultdict(float) 

    print "Smoothing spatial relations.\n"
    for point in points:
        neighbours = get_neighbours(point)
        for neighbour in neighbours:
	    d = (neighbour[0] - point[0], neighbour[1] - point[1], \
                neighbour[2] - point[2])

	    for label_point in range(n_classes+1):
	        for label_neighbour in range(n_classes+1): 
		    p = np.mean([ spatial_rel[(x, (x[0] + d[0], x[1] + d[1], \
			x[2] + d[2]), label_point, label_neighbour)] \
                        for x in get_neighbours_2(point)])

		    spatial_rel_smoothed[(point, neighbour, label_point, \
                        label_neighbour)] = p
		    spatial_rel_normalization[(point, neighbour, label_point)] \
                        += p
    
    # Normalize spatial relations
    print "Normalizing spatial relations.\n"
    for point in points:
        neighbours = get_neighbours(point)
        for neighbour in neighbours:
	    for label_point in range(n_classes+1):
	        for label_neighbour in range(n_classes+1): 
		    p = spatial_rel_normalization[(point, neighbour, \
                        label_point)]

                    if p > 0:
	   	        spatial_rel_smoothed[ (point, neighbour, label_point, \
				label_neighbour) ] /= p
		    else:
	   	        spatial_rel_smoothed[ (point, neighbour, label_point, \
				label_neighbour) ] = -1
			

    with open(spatial_rel_file, 'wb') as fp:
        pickle.dump(spatial_rel_smoothed, fp)
           
def compute_regularization(points, spatial_rel, segmentation, shape):
    """
    Computes regularization using the spatial relations.
    
    Parameters
    ----------
    points : list
        List with the coordinates of voxels.

    spatial_rel_file : string 
        File name for the pickle file to store the spatial relations as a dict.
    The dict is indexed as: 
    (center coords, neighbour coords, center class, neighbour class):probability 
         
    segmentation : ndarray
         Segmentation image.
         
    shape : tuple
         Shape of the priors image (=shape of the segmentation x n_classes).         
          
    Returns
    ----------
    scores : dict
        Dictionary with the scores of the subjects.
    """

    n_labels = shape[3]

    # Add background class (index 0)
    shape = (shape[0], shape[1], shape[2], shape[3] + 1)

    # Compute spatial regularization based on spatial relations
    regularization = np.ones(shape)
    reg_exponent = np.zeros(shape)

    print "Computing regularization using spatial relations.\n"
    for lab_ind in range(n_labels + 1):    
        for point in points:
            neighbours = get_neighbours(point)
            
            for neighbour in neighbours:
                index = (neighbour, point, 
			segmentation[neighbour[0], neighbour[1], neighbour[2]], 
			lab_ind)
                if spatial_rel[index] >= 0:
                    regularization[point[0], point[1], point[2], 
                        lab_ind] *= spatial_rel[index]
                    reg_exponent[point[0], point[1], point[2], 
                        lab_ind] += 1
#		else:
#                    print "Spatial relation not found: %s"%(index,) 
 
    regularization[reg_exponent>0] = np.power(regularization[reg_exponent>0], 
        1./reg_exponent[reg_exponent>0])
    regularization = regularization[:, :, :, 1:]/ \
	np.repeat(np.sum(regularization, axis=3, keepdims=True) + EPS, n_labels, 
        axis=3)

    return(regularization)
    
def do_segmentation(parametric_file, priors_file, mask_file, segmentation_file, 
    segmentation4D_file, spatial_rel_file):
    """
    Do the segmentation based on the priors file and expectation maximization 
    for the probabilities and using regularization.
    
    Parameters
    ----------
    parametric_file : string
        File name for the parametric image.
    
    priors_file : string 
        File name for the priors.
         
    mask_file : string
        File name for mask image.
          
    segmentation_file : string
        File name for segmentation, 1 volume, each class with a different value.
    If equal to '' no regularization is used.
    
    segmentation4D_file : string
        File name for segmentation, 1 volume per class.
    
    spatial_rel_file : string
        File name for the pickle file to store the spatial relations as a dict.
          
    """
    # Load data
    priors_image = load_image(priors_file)
    priors = priors_image.get_data()
    
    orig_priors = priors
    mask_image = load_image(mask_file)
    mask = mask_image.get_data()>0
    parametric = load_image(parametric_file).get_data()[mask]

    n_voxels = np.sum(mask>0)
    n_labels = priors.shape[3]

    mask4D = np.reshape(mask, mask.shape + (1, ) )
    mask4D = np.repeat(mask4D, n_labels, axis=3)
    
    coords = np.where(mask)
    points = set(zip(coords[0], coords[1], coords[2]))

    n_classes = n_labels + 1 
    parametric_matrix = np.repeat(parametric[:, np.newaxis], n_classes, axis=1)    


    if spatial_rel_file == '':
        # Not using regularization    
        print ("Segmenting image.")

        priors_matrix = np.reshape(priors[mask4D], (n_voxels, n_labels))    
	
        # Add background classes
        background = (1 - np.sum(priors_matrix, axis=1, keepdims=True))
        priors_matrix = np.hstack((priors_matrix, background))

        # Expectation-maximization    
        pis = expectation_maximization(parametric_matrix, priors_matrix)

        # Get segmentation
        posteriors = np.zeros(priors.shape)
        posteriors[mask4D] = pis[:, :-1].ravel()
    
        max_pis = np.max(pis, axis=1, keepdims=True)
        labels = 1*(pis == np.repeat(max_pis, n_classes, axis=1))
    
        segmentation4D = np.zeros(priors.shape, dtype=np.int8)
        segmentation4D[mask4D] = labels[:, :-1].ravel().astype(int)

        segmentation = np.zeros(mask.shape, dtype=np.int8)
        for i in range(n_labels):
            segmentation = segmentation + (i+1)*segmentation4D[:, :, :, i]
        
    else: 
        # Using regularization
        print ("Segmenting image with regularization.")

        with open(spatial_rel_file, 'rb') as fp:
            spatial_rel = pickle.load(fp)
        
        # Iterate between expectation maximization and spatial regularization
        segmentation_old = np.zeros(mask.shape, dtype=np.int8)
        seg_diff =  MAX_VOXELS_ICM
    
        iteration = 0

        while (seg_diff >= MAX_VOXELS_ICM and iteration < MAX_ITERS_ICM): 
            print("Iteration %d"%(iteration))
           
            priors_matrix = np.reshape(priors[mask4D], (n_voxels, n_labels))    

            # Add background class
            background = (1 - np.sum(priors_matrix, axis=1, keepdims=True))
            priors_matrix = np.hstack((priors_matrix, background))
            
            # Expectation-maximization
            pis = expectation_maximization(parametric_matrix, priors_matrix)

            # Get posteriors and segmentation
            posteriors = np.zeros(priors.shape)
            posteriors[mask4D] = pis[:, :-1].ravel()
    
            max_pis = np.max(pis, axis=1, keepdims=True)
            labels = 1*(pis == np.repeat(max_pis, n_classes, axis=1))
    
            segmentation4D = np.zeros(priors.shape, dtype=np.int8)
            segmentation4D[mask4D] = labels[:, :-1].ravel().astype(int)

            segmentation = np.zeros(mask.shape, dtype=np.int8)
            for i in range(n_labels):
                segmentation = segmentation + (i+1)*segmentation4D[:, :, :, i]

            # Compute regularization
            regularization = compute_regularization(points, spatial_rel, 
                segmentation, priors.shape)

	    regularization = np.concatenate((1 - np.sum(regularization, axis=3, 
		keepdims=True), regularization), axis=3)
	    posteriors = np.concatenate((1 - np.sum(posteriors, axis=3, 
		keepdims=True), posteriors), axis=3)
            priors = posteriors * regularization
	    priors = priors[:, :, :, 1:]/ \
	        np.repeat(np.sum(priors, axis=3, keepdims=True) + EPS, n_labels, 
                axis=3)
            
	    seg_diff = np.sum(np.abs(segmentation_old - segmentation)>0)

            print("%d voxels changed."%(seg_diff))

            segmentation_old = segmentation   
            iteration += 1    
            
    # Remove spatial relations
    try:
        os.remove(spatial_rel_file)
    except OSError:
        pass 

    # Write out the results
    write_image(segmentation, mask_image.coordmap, segmentation_file)  
    write_image(segmentation4D, priors_image.coordmap, segmentation4D_file)

    
def compute_scores(segmentation_file, target_label_file, score_names):
    """
    Computes summary scores to evaluate the segmentations.
    
    Parameters
    ----------
    segmentation_file : string
        File name for the automatically segmented label image.
    
    target_label_file : string 
        File name for the manually segmented label image.
         
    score_names : list
         List of scores to return.        
          
    Returns
    ----------        
     
    scores : dict
        Dictionary with the scores of the subjects.          
        
    """

    target_labels_image = load_image(target_label_file)
    voxel_size = np.array(target_labels_image.header.get_zooms())
    voxel_vol = voxel_size[0]*voxel_size[1]*voxel_size[2]
    target_labels = target_labels_image.get_data()>0   
    segmentation = load_image(segmentation_file).get_data()>0
    
    scores = {
        'dice': list(),
        'difference': list(),
        'manual_vol': list(),
        'automated_vol': list()
        }
        
    # Compute the scores for all the subjects in the list    
    for i in range(segmentation.shape[3]):
        target_labels_vol = np.sum(target_labels[:,:,:,i])
        segmentation_vol = np.sum(segmentation[:,:,:,i])

        if 'dice' in score_names:
            dice = 2*np.sum(target_labels[:,:,:,i]*segmentation[:,:,:,i])/\
                (target_labels_vol + segmentation_vol)
            scores['dice'].append(str(dice))

        if 'difference' in score_names:
            d1 = np.sum((1*target_labels[:,:,:,i]-1*segmentation[:,:,:,i])>0)
            d2 = np.sum((-1*target_labels[:,:,:,i]+1*segmentation[:,:,:,i])>0)
            difference = (d1 + d2)/(target_labels_vol + segmentation_vol)
            scores['difference'].append(str(difference))

        if 'manual_vol' in score_names:
            manual_vol = np.sum(target_labels[:,:,:,i])*voxel_vol
            scores['manual_vol'].append(str(manual_vol))

        if 'automated_vol' in score_names:
            automated_vol = np.sum(segmentation[:,:,:,i])*voxel_vol
            scores['automated_vol'].append(str(automated_vol))
    
    return(scores)    
