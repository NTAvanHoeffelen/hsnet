r""" UMC inhouse few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import json
from natsort import natsorted
import glob
from medpy import io

class DatasetUMC(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = split #'val' if split in ['val', 'test'] else 'trn' 
        self.fold = fold
        self.nfolds = 1                     # 
        self.nclass = 86                    #
        self.benchmark = 'umc'           
        self.shot = shot
        self.use_original_imgsize = True    # Should probably always be true

        self.img_path = os.path.join(datapath, 'Scans/')            
        self.ann_path = os.path.join(datapath, 'Annotations/')
        self.datapath = datapath
        self.transform = transform                                                  # NOTE: should it contain data augmentation? --> test first without then with

        self.slice_per_class_treshold = 1000

        self.class_ids_train, self.class_ids_val, self.class_ids_test = self.build_class_ids()
        
        self.hsnet_slice_record = self.load_database_json()

        # TODO: A split should be made for test support scans and test query scans.
        # NOTE: This actually should not be necessary. We just need to make sure that when we test on the test set, that the query and support item do not come from the same scan.
        self.training_scans, self.test_scans, self.hsnet_fs_scan_data = self.build_scan_ids()

    # TODO: What is this supposed to be??
    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    # TODO: Edit this to work like i want it to
    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample
    

    def build_class_ids(self):

        class_ids_test = [[1,2],     # [lung_upper_lobe_left, lung_lower_lobe_left] --> Left lung
                           [13],     # aorta
                           [20],     # autochthon_right
                           [22],     # clavicula_right
                           [24],     # duodenum
                           [23],     # heart_atrium_left
                           [31],     # iliac_artery_right
                           [11],     # liver
                           [42, 54], # [rib_left_5, rib_right_5] --> Ribs 5
                           [12]]     # stomach

        class_ids_test_flat = [item for sublist in class_ids_test for item in sublist]

        all_class_ids = np.arange(1,86)

        # NOTE: Should we select these randomly???
        #NR_VAL_CLASSES = 8
        #class_ids_val = np.random.choice(np.setdiff1d(all_class_ids, class_ids_test_flat), NR_VAL_CLASSES, False)
        
        # 8 classes randomly picked
        class_ids_val = [[16],   # pancreas
                         [44],   # rib_left_7
                         [52],   # rib_right_3
                         [8],    # kidney_left
                         [33],   # iliac_vena_right
                         [67],   # vertebrae_C5
                         [76],   # vertebrae_T10
                         [71]]   # vertebrae_L2
        
        class_ids_val_flat = [item for sublist in class_ids_val for item in sublist]

        class_ids_train = np.setdiff1d(np.setdiff1d(all_class_ids, class_ids_test_flat), class_ids_val_flat)
        class_ids_train = [[item] for item in class_ids_train]

        return class_ids_train, class_ids_val, class_ids_test

    def build_scan_ids(self):
        try:
            hsnet_fs_scan_data = open(os.path.join(self.datapath, "/HSnet_umc_scans_split.json"))
            test_scans = hsnet_fs_scan_data['test_scans']
            training_scans = hsnet_fs_scan_data['train_and_val_scans']

            return training_scans, test_scans, hsnet_fs_scan_data
        except:
            print(f"ran into an issue when trying to open HSnet_umc_scans_split.json")

        nr_scans = self.database_info["nr_scans"]

        test_scan_percentage = 0.2

        test_amount = int(np.floor(nr_scans * test_scan_percentage))
        train_amount = int(np.ceil(nr_scans * (1 - test_scan_percentage)))

        scan_split = False

        while not scan_split:

            # Randomly select test and train/val scans
            selected_test_scans = np.random.choice(np.arange(0,nr_scans), test_amount, False, None)
            selected_training_scans = np.random.choice(np.setdiff1d(np.arange(0,nr_scans), selected_test_scans), train_amount, False)

            # loop over classes in training and validation
            # and get the number of positive slices ih the selected scans
            nr_slices_per_class = {}
            for class_ in self.class_ids_train + self.class_ids_val:
                nr_slices_per_class[str(class_)] = 0
                for scan_ in selected_training_scans:
                    nr_slices_per_class[str(class_)] += self.database_info["slice_record"][str(class_)][str(scan_)]['nr_pos_slices']
            
            # loop over classes in test:
            # and get the number of positive slices ih the selected scans
            for class_ in self.class_ids_test:
                nr_slices_per_class[str(class_)] = 0
                for scan_ in selected_test_scans:
                    nr_slices_per_class[str(class_)] += self.database_info["slice_record"][str(class_)][str(scan_)]['nr_pos_slices']
            
            # Check if there are enough slices per class
            if sum(np.array(list(nr_slices_per_class.values())) > self.slice_per_class_treshold) == len(nr_slices_per_class.values()):

                # if so, write info to the json file
                hsnet_fs_scan_data = {}
                hsnet_fs_scan_data['train_and_val_scans'] = selected_training_scans
                hsnet_fs_scan_data['test_scans'] = selected_test_scans
                hsnet_fs_scan_data['train_classes'] = self.class_ids_train
                hsnet_fs_scan_data['val_classes'] = self.class_ids_val
                hsnet_fs_scan_data['test_classes'] = self.class_ids_test
                hsnet_fs_scan_data['nr_pos_slices'] = nr_slices_per_class

                with open(self.datapath + f"/HSnet_umc_scans_split.json", 'w') as f:
                    json.dump(hsnet_fs_scan_data, f, indent = 4)

                scan_split = True
        return selected_training_scans, selected_test_scans, hsnet_fs_scan_data


    def load_database_json(self):
        try:
            # Open json
            database_info_file = open(os.path.join(self.datapath, "/HSnet_slice_record.json"))

            # Load json
            hsnet_slice_record = json.load(database_info_file)
        except:
            print("Could not load/find HSnet_slice_record.json")
            print("Continueing by generating a new HSnet_slice_record.json...")
            self.create_fs_slice_json()

            # Open json
            database_info_file = open(os.path.join(self.datapath, "/HSnet_slice_record.json"))

            # Load json
            hsnet_slice_record = json.load(database_info_file)

        return hsnet_slice_record
    
    def create_fs_slice_json(self):
                
        # Get path of images and annotations
        list_of_img_files = natsorted(glob.glob(self.datapath + "Scans/*.nii.gz"))
        list_of_annot_files = natsorted(glob.glob(self.datapath + "Annotations/*.nii.gz"))

        # Keeps track of the positive and negative slices per scan
        slice_record = {}

        # Keeps track of which scans have 0 non-class slices
        train_scan_without_non_class_slices = []

        # add together
        combined_classes = self.class_ids_test + self.class_ids_val + self.class_ids_train

        # Find the slices with and without the class per training scan
        for scan_idx in range(0, len(list_of_img_files)):

            # load scan
            image_data, _, annot_data, _ = self.__load_scan_and_annotation_dirs__(list_of_img_files, list_of_annot_files, scan_idx)

            # loop over all classes
            for class_id in combined_classes:

                image_data_copy, annot_data_copy = image_data.copy(), annot_data.copy()

                if str(class_id) not in slice_record.keys():
                    slice_record[str(class_id)] = {}
                if scan_idx not in slice_record[str(class_id)].keys():
                    slice_record[str(class_id)][scan_idx] = {}


                annot_data = self.__remove_classes__(annot_data_copy, class_id)

                # Get all slices containing foreground class
                class_slices, non_class_slices = self.__get_class_slices__(image_data_copy, annot_data_copy)

                if len(list(non_class_slices)) == 0:
                    train_scan_without_non_class_slices.append(str(scan_idx))

                slice_record[str(class_id)][scan_idx]['positive_cases'] = list(class_slices)
                slice_record[str(class_id)][scan_idx]['negative_cases'] = list(non_class_slices)
                slice_record[str(class_id)][scan_idx]['nr_pos_slices'] = len(list(class_slices))
            
        # Write info to the json file
        new_class_data = {}
        new_class_data['nr_scans'] = len(list_of_img_files)
        new_class_data['slice_record'] = [slice_record]

        with open(self.datapath + f"/HSnet_slice_record.json", 'w') as f:
            json.dump(new_class_data, f, indent = 4)

    def __load_scan_and_annotation_dirs__(list_of_img_files, list_of_annot_files, scan_index):
        ''' Load a scan and annotation file picked from a pandas data frame at {scan_index} '''
        
        # Check if both the scan and annotation exist
        scan_exists = os.path.exists(list_of_img_files[scan_index])
        annotation_exists = os.path.exists(list_of_annot_files[scan_index])

        # if both scan and its annotation are found
        if scan_exists and annotation_exists:

            # load image and annotation
            image_data, image_header = io.load(list_of_img_files[scan_index])
            annotation_data, annotation_header = io.load(list_of_annot_files[scan_index])

            return image_data, image_header, annotation_data, annotation_header

    def __remove_classes__(self, annot_data, class_val):
        ''' Set the class label of specified classes to 1 and remove the class labels of not specified classes'''
        # When we dont merge multiple classes
        if len(class_val) == 1:
            annot_data[annot_data != class_val[0]] = 0 # set all but the chosen class to 0
            annot_data[annot_data == class_val[0]] = 1 # set chosen class to 1
        else:
            if 1 in class_val: 
                for cv in class_val:                    # set each class to foreground
                    annot_data[annot_data == cv] = 1
                annot_data[annot_data != 1] = 0         
            else: 
                annot_data[annot_data == 1] = 0 # if 1 is not in the list of classes we need to set it to 0 first
                for cv in class_val:
                    annot_data[annot_data == cv] = 1
                annot_data[annot_data != 1] = 0
        return annot_data
    
    def __get_class_slices__(self, image_data, annotation_data):
        ''' Find the slices which contain the foreground class '''

        class_slices = []
        non_class_slices = []

        # loop through all slices and find which hold class labels
        for slice_nr in range(0, annotation_data.shape[2]):
                # check if slice contains class of interest:
                if 1 in np.unique(annotation_data[:,:,slice_nr]) and len(np.unique(image_data[:,:,slice_nr])) > 1:
                    
                    # save slice nr in list
                    class_slices.append(slice_nr)

                elif 0 in np.unique(annotation_data[:,:,slice_nr]) and len(np.unique(image_data[:,:,slice_nr])) > 1:
                    non_class_slices.append(slice_nr)

        return class_slices, non_class_slices