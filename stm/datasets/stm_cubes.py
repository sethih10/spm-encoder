"""STM z-stacks without the molecule, for the reconstruction arm.

The contrastive arm needs (cube, molecule) pairs; reconstruction needs only the cube. Reading
the geometry anyway would cost an extra HDF5 read per sample and force a custom collate for
the ragged atom arrays, so this drops both.
"""

from .stm_pairs import StmPairDataset


class StmCubeDataset(StmPairDataset):
    """Emits ``{'cube': (Z, H, W)}``: ``Z`` z-slices of an ``H`` x ``W`` image."""

    @property
    def height(self):
        return self.image_size

    @property
    def width(self):
        return self.image_size

    def __getitem__(self, idx):
        return {'cube': super().__getitem__(idx)['cube']}
