import os
import re
from pathlib import Path

from PIL import Image
from torchvision.datasets import VisionDataset
import numpy as np


#  800000	Bareland	1.5
#  00FF24	Rangeland	22.9
#  949494	Developed space	16.1
#  FFFFFF	Road	6.7
#  226126	Tree	20.2
#  0045FF	Water	3.3
#  4BB549	Agriculture land	13.7
#  DE1F07	Building
class OEMSemanticDataset(VisionDataset):

    def __init__(self, metainfo, dataset_dir, transform, target_transform=None,
                 image_set='train',
                 img_suffix='.tif',
                 ann_suffix='.tif',
                 data_prefix: dict = dict(img_path='images', ann_path='labels'),
                 return_dict=False):
        '''

        :param metainfo: meta data in original dataset, e.g. class_names
        :param dataset_dir: the path of your dataset, e.g. data/my_dataset/ by the stucture tree above
        :param image_set: 'train' or 'val'
        :param img_suffix: your image suffix
        :param ann_suffix: your annotation suffix
        :param data_prefix: data folder name, as the tree shows above, the data_prefix of my_dataset: img_path='img' , ann_path='ann'
        :param return_dict: return dict() or tuple(img, ann)
        '''
        super(OEMSemanticDataset, self).__init__(root=dataset_dir, transform=transform,
                                                  target_transform=target_transform)

        self.class_names = metainfo['class_names']
        self.dataset_dir = dataset_dir
        self.img_names = [image_id.strip() for image_id in open(Path(dataset_dir) / f'{image_set}.txt').readlines()]
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.return_dict = return_dict

    def _get_image_dir(self, image_id):
        # get folder name from image_id, e.g. san_tome_10.tif -> san_tome/
        dir_name = re.findall(r'(.+?)_\d+.tif', image_id)[0] + '/'
        return self.dataset_dir + dir_name

    def __getitem__(self, index):
        img_dir = self._get_image_dir(self.img_names[index])
        img = Image.open(os.path.join(img_dir, 'images', self.img_names[index]))
        ann = Image.open(os.path.join(img_dir, 'labels', self.img_names[index]))
        if self.transforms is not None:
            img, ann = self.transforms(img, ann)
        ann = np.array(ann)

        if self.return_dict:
            data = dict(img_name=self.img_names[index], img=img, ann=ann,
                        img_path=os.path.join(img_dir, 'images', self.img_names[index]),
                        ann_path=os.path.join(img_dir, 'labels', self.img_names[index]))
            return data
        return img, ann

    def __len__(self):
        return len(self.img_names)
