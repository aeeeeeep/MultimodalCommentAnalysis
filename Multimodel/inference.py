import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from model import MultiModal
from tqdm import tqdm


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, "../tools/label.csv", test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with open(args.test_output_csv, 'w') as f:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                asin, time, pred = model(batch, infer=True)
                for i in range(len(asin)):
                    f.write(f'{asin[i]},{time[i]},{pred[i]}\n')

if __name__ == '__main__':
    inference()
