#!/usr/bin/python
"""
usage: MBST.py [-h] [-c] [-o] tg_struct tg_brain tg_param atlas output

MidBrain Segmentation Tool: Automated segmentation of the red nucleus,
substantia nigra and subthalamic nucleus.

positional arguments:
  tg_struct   Target structural file.
  tg_brain    Brain-extracted target structural file.
  tg_param    Parametric file.
  atlas       Text file containing the atlas.
  output      Output directory.

optional arguments:
  -h, --help  show this help message and exit
  -c          Clean up files after registration (default=TRUE).
  -o          Overwrite registration files (default=FALSE).

"""
# Author: Benjamin Garzon <benjamin.garzon@gmail.com>
# License: BSD 3 clause

import sys
import argparse
from segment import register, estimate_spatial_relations, do_segmentation, \
fuse_labels

def MBST(args):

    target_structural_file = args.tg_struct
    target_structural_brain_file = args.tg_brain
    target_parametric_file = args.tg_param
    atlas_file = args.atlas
    output_dir = args.output
    clean_up = args.c
    overwrite = args.o

    priors_file = '%s/priors.nii.gz'%(output_dir)    
    average_mask_file = '%s/average_mask.nii.gz'%(output_dir)
    
    atlas_subjects = list()
    f_atlas = open(atlas_file, 'r')    
    for line in f_atlas:
        subject, structural_file, structural_brain_file, \
            structural_mask_file, i_file, m_file, \
            label_file = line.replace('\n', '').split(':')
        atlas_subjects.append(subject)
    f_atlas.close()
    
    register(target_structural_file, target_structural_brain_file, 
        target_parametric_file, atlas_file, output_dir, clean_up, overwrite)
    
    fuse_labels(atlas_subjects, output_dir, priors_file, average_mask_file)
            
    estimate_spatial_relations(atlas_subjects, output_dir, spatial_rel_file, 
        average_mask_file)        
            
    do_segmentation(atlas_subjects, priors_file, average_mask_file, 
        segmentation_file, segmentation4D_file, spatial_rel_file)
        
def main():

    parser = argparse.ArgumentParser(description='MidBrain Segmentation Tool: \
    Automated segmentation of the red nucleus, substantia nigra and \
    subthalamic nucleus.')
    
    parser.add_argument('tg_struct',
                   help='Target structural file.')
                   
    parser.add_argument('tg_brain',
                   help='Brain-extracted target structural file.') 
                   
    parser.add_argument('tg_param',
                   help='Parametric file.') 

    parser.add_argument('atlas',
                   help='Text file containing the atlas.')
                   
    parser.add_argument('output',
                   help='Output directory.') 

    parser.add_argument("-c", help="Clean up files after \
        registration (default=TRUE).", action="store_false")                                               
                    
    parser.add_argument("-o", help="Overwrite registration files \
        (default=FALSE).", action="store_true")                                               
                    
    MBST(parser.parse_args())

if __name__ == "__main__":
    main()	        
        
