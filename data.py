import torch

def create_dataset(opt, data):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt, data)
    dataset = data_loader.load_data()
    return dataset

class SimplificationDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return data

class CustomDatasetDataLoader():
    """Wrapper class to load the dataset"""

    def __init__(self, opt, data):
        self.opt = opt
        self.data = data
        # model_name = "facebook/bart-large-cnn"
        # self.tokenizer = BartTokenizer.from_pretrained(model_name)
        # tokenized_train_datasets = self.data.map(self.preprocess_function, batched=True, batch_size=opt.batch_size)
        self.dataset = SimplificationDataset(self.data)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data