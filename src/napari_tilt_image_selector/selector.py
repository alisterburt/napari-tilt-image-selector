from os import PathLike
from typing import List, Optional

import napari
import numpy as np
from napari.utils.events.containers import SelectableEventedList

from .utils import get_ordered_tilt_images, lazy_tilt_series_from_tilt_images


class Selector:
    def __init__(self, viewer: napari.Viewer):
        self.mdoc_files: Optional[SelectableEventedList[PathLike]] = None
        self.viewer = viewer

        dummy_image = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.image_layer = self.viewer.add_image(data=dummy_image)

    @property
    def mdoc_files(self):
        return self._mdoc_files

    @mdoc_files.setter
    def mdoc_files(self, value: List[PathLike]):
        self._mdoc_files = SelectableEventedList(value)

    @property
    def tilt_image_files(self):
        return self._tilt_image_files

    @tilt_image_files.setter
    def tilt_image_files(self, value: List[PathLike]):
        self._tilt_image_files = value

    def load_tilt_series(self, mdoc_file: PathLike):
        ordered_tilt_images = get_ordered_tilt_images(
            mdoc_file=mdoc_file, tilt_image_files=self.tilt_image_files
        )
        tilt_series = lazy_tilt_series_from_tilt_images(ordered_tilt_images)
        self.image_layer.data = tilt_series

    def _update_image_layer(self):
        """Update image layer from current micrograph"""
        self.image_layer.data = self.current_micrograph.image_data
        self.image_layer.contrast_limits_range = (-2, 2)
        self.image_layer.reset_contrast_limits()
        self.viewer.reset_view()
