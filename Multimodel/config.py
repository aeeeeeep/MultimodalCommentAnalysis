import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2023, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_data_file', type=str, default='data/train.csv')
    parser.add_argument('--val_data_file', type=str, default='data/val.csv')
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float,
                        help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=48, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=128, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=128, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=15, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/v1')
    parser.add_argument('--ckpt_file', type=str, default='save/v1/model_epoch_1_mean_f1_0.9488.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=2, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=5, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== text BERT =============================
    parser.add_argument('--bert_dir', type=str, default='bert-base-uncased')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str,
                        default='/root/autodl-tmp/checkpoint-muti/swin_tiny_patch4_window7_224_22k.pth')

    # ========================== Video =============================
    parser.add_argument('--image_embedding_size', type=int, default=768)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_hidden_size', type=int, default=768, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    return parser.parse_args()
