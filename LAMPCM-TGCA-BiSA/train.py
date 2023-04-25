import argparse
from train_val import *
from sklearn.model_selection import KFold
from pathlib import Path
import sys
sys.path.append(str(Path('..').absolute()))
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', default=256, type=int, required=False,
                        #  help='..')
    parser.add_argument('--batch_size', default=32, type=int, required=False,
                         help='..')
    # parser.add_argument('--epochs', default=2000, type=int, required=False,
    #                      help='..')
    parser.add_argument('--epochs', default=100, type=int, required=False,
                         help='..')
    parser.add_argument('--lr', default=0.001, type=float, required=False,
                         help='..')
    # parser.add_argument('--lr', default=1e-4, type=float, required=False,
    #                      help='..')
    parser.add_argument('--wd', default=5e-2, type=float, required=False,
                         help='..')
    parser.add_argument('--dataset', default='davis', type=str, required=False,
                        help='..')
    parser.add_argument('--gpu', type=int, default=1, help='n-th cuda GPU, from 0 to 7')

    parser.add_argument('--fold', type=int, default=0, help='5-fold cross validation')
    parser.add_argument('--cutoff', type=int, default=None, help='cutoff for test dataset')

    args = parser.parse_args()
    model = LAMPCM()


    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.0005, last_epoch=-1)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.wd
    )
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=args.epochs,
        max_lr=args.lr,
        min_lr=1e-8,
        warmup_steps=int(args.epochs * 0.1)
    )

    train_dataset, valid_dataset = get_train_valid(dataset=args.dataset , fold=args.fold, cutoff=args.cutoff) # D
    test_dataset = get_test(dataset=args.dataset, cutoff=args.cutoff) # D
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate) # FIXME
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate) # FIXME
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate)
    train_eval(model, optimizer,scheduler, train_loader, val_loader, test_loader, args.epochs, args.dataset, gpu=args.gpu, fold=args.fold, test_cutoff=args.cutoff)


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    main()
