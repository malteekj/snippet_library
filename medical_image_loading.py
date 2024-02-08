#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:49:04 2023

@author: maltejensen
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import pydicom
import os 
import numpy as np
import time
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import SimpleITK as sitk

def sitkDicomReader(dcm_path, threshold=None, flip=None, return_array = False):
    '''
    Wrapper function that makes dicom reading simple with SimpleITK
    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Added a call to PermuteAxes to change the axes of the data
    # image = sitk.PermuteAxes(image, [2, 1, 0])
    
    if threshold is not None:
        image = sitk.BinaryThreshold(image,lowerThreshold=threshold[0], upperThreshold=threshold[1])
    
    if flip is not None:
        image = sitk.Flip(image, flip)
    
    if return_array:
        return sitk.GetArrayFromImage(image)
    else:
        return image

def sitkDicom2Nifti(dcm_path, out_path, threshold=None, flip=None):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Added a call to PermuteAxes to change the axes of the data
    # image = sitk.PermuteAxes(image, [2, 1, 0])
    
    if threshold is not None:
        image = sitk.BinaryThreshold(image,lowerThreshold=threshold[0], upperThreshold=threshold[1])
    
    if flip is not None:
        image = sitk.Flip(image, flip)
    
    sitk.WriteImage(image, out_path)
    
    return image

def readAndResample(CT_path, PET_path):
    '''
    Function that read and resample pet 
    '''
    print('reading PET..')
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(PET_path)
    reader.SetFileNames(dicom_names)
    
    reader.MetaDataDictionaryArrayUpdateOn()
    # reader.LoadPrivateTagsOn()
    reader.SetOutputPixelType(sitk.sitkFloat32)
    imagePET = reader.Execute()
    
    print('reading CT..')
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(CT_path)
    reader.SetFileNames(dicom_names)
    
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    imageCT = reader.Execute()
    
    print('resampling PET..')
    transform = sitk.Transform(3, sitk.sitkIdentity)
    # interpolation = sitk.sitkNearestNeighbor
    interpolation = sitk.sitkLinear
    # interpolation = sitk.sitkBSpline
    # interpolation = sitk.sitkCosineWindowedSinc
    
    
    t0 = time.time()
    pet_resampled = sitk.GetArrayFromImage(sitk.Resample(imagePET, imageCT, transform, interpolation)) #.astype(np.int32)
    pet_resampled = np.transpose(pet_resampled, axes=[1,2,0])
    print('resampling time: {:.2f} s'.format(time.time()-t0))
    
    ct_array = sitk.GetArrayFromImage(imageCT) #.astype(np.int32)
    ct_array = np.transpose(ct_array, axes=[1,2,0]) #.astype(np.int32)
    
    pet_array = sitk.GetArrayFromImage(imagePET) #.astype(np.int32)
    pet_array = np.transpose(pet_array, axes=[1,2,0]) #.astype(np.int32)
    
    return pet_resampled, pet_array, ct_array


class multi_slice_viewer:
    
    color_array = plt.get_cmap('jet')(range(256))
    color_array[:4,-3:] = 0
    map_object_jet = LinearSegmentedColormap.from_list(name='segmented_colormap',colors=color_array)


    def __init__(self,volume, second_volume=None, second_cmap='hot', alpha=0.5, rois=None,
                 start_slice=0, start_frame=0, shift_correct=[0,0], window=[None, None], second_window=[None, None],
                 cmap='gray', orientation='axial', show_tool_tips=False, title=None, blocking=False, threshold_mode=False):
        
        if cmap == 'seg':
            # create transparent colormaps for ROIs
            if None in window:
                window[0] = int(volume.min())
                window[1] = int(volume.max())
            
            color_array = plt.get_cmap('Set1')(range(window[1] - window[0] + 1))
            color_array = np.concatenate( (np.array([[0,0,0,0]]),color_array[:-1,:]), axis=0)
            map_object_seg = ListedColormap(name='segmentation_cmap',colors=color_array)
    
            cmap = map_object_seg
        elif cmap == 'prop':
            cmap = self.map_object_jet
        else:
            if None in window:
                window[0] = int(volume.min())
                window[1] = int(volume.max())
            
            
       
        if second_cmap == 'seg':
            if None in second_window:
                second_window[0] = int(second_volume.min())
                second_window[1] = int(second_volume.max())
            
            self.second_interp = 'nearest' 
            
            color_array = plt.get_cmap('Set1')(range(second_window[1] - second_window[0] +1 ))
            color_array = np.concatenate( (np.array([[0,0,0,0]]),color_array[:-1,:]), axis=0)
            map_object_seg = ListedColormap(name='segmentation_cmap',colors=color_array)
            
            second_cmap  = map_object_seg
            
        elif second_cmap == 'prop':
            second_cmap = self.map_object_jet
            self.second_interp = 'antialiased'
        else:
            self.second_interp = 'antialiased'
        
        
        self.rois = rois
        self.remove_keymap_conflicts({'up', 'down','j','l','q','a','d'})
        self.fig, self.ax = plt.subplots(figsize=(10,7.5))
        if title is not None:
            self.fig.suptitle(title)
            
        self.second_volume = second_volume
        self.orientation = orientation
        self.threshold_mode = threshold_mode
        
        # expand with dummy dimension for 3D data
        if len(volume.shape) == 3:
            volume = np.expand_dims(volume, axis=3)
            self.ax.total_frames = 1
            self.ax.frame = 0
        elif len(volume.shape) == 4:
            self.ax.total_frames = volume.shape[3]
            self.ax.frame = start_frame 
        
            
        # handle the orientation of the image
        if orientation == 'axial':
            self.ax.volume = volume
            if second_volume is not None:
                self.ax.sec_volume = second_volume
        elif orientation == 'sagittal':
            self.ax.volume = np.transpose(volume, (0,2,1,3))
            if second_volume is not None:
                self.ax.sec_volume = np.transpose(second_volume, (0,2,1))
        elif orientation == 'coronal':
            self.ax.volume = np.transpose(volume, (2,1,0,3))
            if second_volume is not None:
                self.ax.sec_volume = np.transpose(second_volume, (2,1,0))
        else:
            raise KeyError('Orientation has to be either: \'axial\', \'sagittal\' or \'coronal\'') 
            
        self.ax.rois = rois
        self.ax.index = start_slice
        self.frame_mode = False
        self.slice_jump = 1
       
        # for testing correction
        self.x_shift = shift_correct[0]
        self.y_shift = shift_correct[1]
        
        # If used for thresholding experiments
        if self.threshold_mode:
            self.threshold = 0.5
            # define binary image
            self.ax.sec_volume_threshold = np.zeros_like(self.ax.sec_volume, dtype=np.int8)
            self.ax.sec_volume_threshold[self.ax.sec_volume >= self.threshold] = 1
         
        
        self.ax.imshow(self.ax.volume[:,:,self.ax.index, self.ax.frame], cmap=cmap, origin='lower', vmin=window[0], vmax=window[1])
        if self.second_volume is not None:
            self.alpha = alpha
            if self.threshold_mode:
                self.ax.imshow(self.ax.sec_volume_threshold[:,:,self.ax.index], cmap=self.map_object_seg, origin='lower', 
                               alpha=self.alpha, vmin=second_window[0], vmax=second_window[1])
            else:
                self.ax.imshow(self.ax.sec_volume[:,:,self.ax.index], cmap=second_cmap, origin='lower', alpha=self.alpha, 
                               vmin=second_window[0], vmax=second_window[1], interpolation=self.second_interp)
            
        self.fig.canvas.mpl_connect('key_release_event', self.process_key)
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.fig.canvas.mpl_connect('scroll_event', self.process_key)
        self.fig.canvas.mpl_connect('button_press_event', self.process_key)
        self.fig.canvas.mpl_connect('button_release_event', self.process_key)
        
        self.ax.set_title(self.ax.index)
        
        self.ax.drawed_roi = []
        if self.ax.rois is not None: 
            self.draw_rois()
         
        self.ax.set_title('slice: {}, frame: {}'.format(self.ax.index, self.ax.frame))
        plt.show()
        
        # if used in a loop, where used in a blocking way
        if blocking:
            self.fig.canvas.start_event_loop()
        
        # print options
        if show_tool_tips:
            self.print_key_options()
            
    def process_key(self, event):
        
        # Release ctrl or shift key
        if event.name == 'key_release_event' and event.key == 'control':
            self.slice_jump = 1
        elif event.name == 'key_release_event' and event.key == 'shift':
            self.frame_mode = False
        
        if event.name == 'key_press_event':
            # Controlling scroll mode
            if event.key == 'control':
                self.slice_jump = 10
            elif event.key == 'shift':
                self.frame_mode = True
            elif 'up' in event.key:
                self.previous_slice()
            elif 'down' in event.key:
                self.next_slice()
            elif event.key == 'j':
                self.decrease_alpha()
            elif event.key == 'l':
                self.increase_alpha()
            elif event.key == 'q':
                self.close_fig()    
            elif event.key == 'a':
                self.decrease_threshold()    
            elif event.key == 'd':
                self.increase_threshold()
                
        elif event.name == 'scroll_event':
            if event.button == 'up':
                self.previous_slice()
            elif event.button == 'down':
                self.next_slice()
        elif event.name == 'button_press_event' and event.button == 3:
            self.start_intensity_window(event)
        elif event.name == 'button_release_event' and event.button == 3:
            self.end_intensity_window(event)
                
        self.ax.set_title('slice: {}, frame: {}'.format(self.ax.index, self.ax.frame))
        self.fig.canvas.draw()
    

    def decrease_alpha(self):
        self.alpha -= 0.05
        if self.alpha < 0:
            self.alpha = 0
        self.ax.images[1].set_alpha(self.alpha)
    
    def increase_alpha(self):
        self.alpha += 0.05
        if self.alpha > 1:
            self.alpha = 1
        self.ax.images[1].set_alpha(self.alpha)
    
    def decrease_threshold(self):
        self.threshold -= 0.05
        if self.threshold < 0:
            self.threshold = 0
        self.update_threshold_volume()
            
    def increase_threshold(self):
        self.threshold += 0.05
        if self.threshold > 1:
            self.threshold = 1
        self.update_threshold_volume()
        
    def update_threshold_volume(self):
        self.ax.sec_volume_threshold.fill(0)
        self.ax.sec_volume_threshold[self.ax.sec_volume >= self.threshold] = 1
        self.draw()
        
    def draw(self):
        self.ax.images[0].set_array(self.ax.volume[:,:,self.ax.index, self.ax.frame])
        # Drawing the second volume 
        if self.second_volume is not None:
            if self.threshold_mode:
                self.ax.images[1].set_array(self.ax.sec_volume_threshold[:,:,self.ax.index])
            else:
                self.ax.images[1].set_array(self.ax.sec_volume[:,:,self.ax.index])
            
        # find roi if any
        for drawing in self.ax.drawed_roi:
            drawing[0].remove()
        
        self.ax.drawed_roi = []
        if self.rois is not None:
            self.draw_rois()
     
    def previous_slice(self):
        if self.frame_mode:
            self.ax.frame = (self.ax.frame-1) % self.ax.total_frames
            # Adjust for different scales in each frame
            self.frame_vmin, self.frame_vmax = self.ax.volume[:,:,self.ax.index, self.ax.frame].min(), self.ax.volume[:,:,self.ax.index, self.ax.frame].max()
            self.ax.images[0].set_clim(vmin=self.frame_vmin, vmax=self.frame_vmax)
        else:
            self.ax.index = (self.ax.index - self.slice_jump) % self.ax.volume.shape[2]  # wrap around using %
        
        self.draw()
            
    def next_slice(self):
        if self.frame_mode:
            self.ax.frame = (self.ax.frame+1) % self.ax.total_frames
            # Adjust for different scales in each frame
            self.frame_vmin, self.frame_vmax = self.ax.volume[:,:,self.ax.index, self.ax.frame].min(), self.ax.volume[:,:,self.ax.index, self.ax.frame].max()
            self.ax.images[0].set_clim(vmin=self.frame_vmin, vmax=self.frame_vmax)
        else:
            self.ax.index = (self.ax.index + self.slice_jump) % self.ax.volume.shape[2]  # wrap around using %
        
        self.draw()
    
    def start_intensity_window(self, event):
        # Record the start of the intensity windows
        self.x_0, self.y_0 = event.xdata, event.ydata

    def end_intensity_window(self, event):
        self.x_1, self.y_1 = event.xdata, event.ydata
        # Sort the values before indexing
        x_min, x_max = np.sort([self.x_0, self.x_1]).astype(int) 
        y_min, y_max = np.sort([self.y_0, self.y_1]).astype(int) 
        # Find min and max within square        
        new_vmin = np.min(self.ax.volume[y_min:y_max, x_min:x_max,self.ax.index, self.ax.frame])
        new_vmax = np.max(self.ax.volume[y_min:y_max, x_min:x_max,self.ax.index, self.ax.frame])
        
        self.ax.images[0].set_clim(vmin = new_vmin, vmax = new_vmax)
          
    def draw_rois(self):
        # Function for drawing the rois
        for roi in self.ax.rois: # a list rois for each structure
             for roi_slice in roi: # each slice in that roi
                 if self.orientation == 'axial':
                     if int(roi_slice[0,2]) == self.ax.index: # draw if z coordinate is the current slice
                         self.ax.drawed_roi.append(self.ax.plot(roi_slice[:,0]+self.x_shift, roi_slice[:,1]+self.y_shift,'r*'))
                 if self.orientation == 'coronal':
                     idx_temp = np.where(roi_slice[:,1] == self.ax.index)[0]
                     self.ax.drawed_roi.append(self.ax.plot(roi_slice[idx_temp,0]+self.x_shift, roi_slice[idx_temp,2]+self.y_shift,'r*'))
                
    def close_fig(self):
        self.fig.canvas.stop_event_loop()
        plt.close(self.fig)
        
    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    
    @staticmethod
    def help():
        print('- Scrolling between slices can be done by scrolling with the mouse or with the up and down button')
        print('- To fast scroll, hold down \'ctrl\' while scrolling')
        print('- To scroll the 4th dimension (different scans), hold down \'shift\' while scrolling')
        print('- To fade second volume in/out use \'j\' and \'l\'')
        print('- Press and hold the right mouse button to adjust the dynamic range to that window')
        print('- Press \'q\' to exit figure')
        print('- Use \'seg\' for segmentation or \'prop\' for ajustable threshold. Use \'a\' and \'d\' to decrease and increase the threshold')
