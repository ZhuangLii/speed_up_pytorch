from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# ----- 1 ----- #
train_loader = DataLoaderX(train_dataset, batch_size, num_worker, ....)
# ----- 2 ----- #
train_loader = DataLoader(train_dataset, batch_size, num_worker, ....)

