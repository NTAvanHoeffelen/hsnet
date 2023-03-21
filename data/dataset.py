r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS
from data.umc import DatasetUMC

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):
        
        # TODO: Add inhouse dataset
        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'umc': DatasetUMC,
        }

        cls.img_mean = [0.485, 0.456, 0.406]  # Probably not needed with grayscale data
        cls.img_std = [0.229, 0.224, 0.225]   # Probably not needed with grayscale data
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        # TODO: Probably not going to resize the inhouse data; Maybe add augmentation?
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
