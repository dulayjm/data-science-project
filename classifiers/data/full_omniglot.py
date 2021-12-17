"""
This file contains the FullOmniglot class,
which is a modification of the Omniglot class in PyTorch save that it works with the full Omniglot dataset,
as the full dataset--for whatever reason--was not initially accessible from there.
Only certain subsets of the Omniglot class were.

The original code can be found here:
https://pytorch.org/vision/stable/datasets.html?highlight=omniglot#torchvision.datasets.Omniglot
"""

from os.path import join
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import list_dir, list_files


class FullOmniglot(VisionDataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists. It must already exist under the name images_full in this directory.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    folder = 'omniglot-py'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super(FullOmniglot, self).__init__(join(root, self.folder), transform=transform,
                                           target_transform=target_transform)

        self._target_folder = 'images_full'
        self.target_folder = join(self.root, self._target_folder)
        self._alphabets = list_dir(self.target_folder)
        self._characters: List[str] = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                           for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])

    def __len__(self) -> int:
        return len(self._flat_character_images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class
