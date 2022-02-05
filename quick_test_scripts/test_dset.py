from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

import torch


class CustomDset(Dataset):

    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 10

    def __getitem__(self, index):
        x = torch.randn((2, 10))
        y = torch.randn((torch.randint(10, size=(1,))[0], 2))

        # cs, ca, ts, ta
        return dict(x=x, y=y)


def custom_collate(batch):
    # Note: we will have dynamic batch size for training the MLP part which should be ok? not sure!
    elem = batch[0]
    ret = {}
    for key in elem:
        if key != 'y':
            ret[key] = torch.stack([e[key] for e in batch], 0)
    
    ret['y'] = torch.cat([e['y'].view(-1, elem['y'].shape[-1]) for e in batch])
    ret['ptr'] = torch.tensor([len(e['y']) for e in batch])

    return ret
    

dset = CustomDset()
loader = DataLoader(dset, batch_size=2, collate_fn=custom_collate)
batch = next(iter(loader))

x = torch.repeat_interleave(batch['x'], batch['ptr'], dim=0)
breakpoint()