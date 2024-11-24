import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model.sasrec import SASRecModel
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix

# Set up arguments
class Args:
    data_dir = "./data/"
    output_dir = "output/"
    data_name = "final"
    do_eval = False
    load_model = None
    train_name = "test_model"
    num_items = 10
    num_users = 10
    lr = 0.001
    batch_size = 512
    epochs = 10
    no_cuda = False
    log_freq = 1
    patience = 5
    num_workers = 0  # Set num_workers to 0 to avoid BrokenPipeError on Windows
    seed = 42
    weight_decay = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    gpu_id = "0,1,2,3"
    variance = 5
    model_type = 'sasrec'
    max_seq_length = 10
    hidden_size = 256
    num_hidden_layers = 2
    hidden_act = "gelu"
    num_attention_heads = 2
    attention_probs_dropout_prob = 0.5
    hidden_dropout_prob = 0.5
    initializer_range = 0.02

args = Args()

if __name__ == "__main__":
    # Initialize logger
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory if not exists
    check_path(args.output_dir)

    # Set CUDA environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # Load data
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    # Prepare checkpoint paths
    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')

    # Load dataloaders
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    # Initialize and log model
    logger.info(str(args))
    model = SASRecModel(args=args)
    logger.info(model)

    # Initialize trainer
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    # Generate rating matrices for evaluation
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    # Training and evaluation
    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)
    
torch.save(model, './llmeb_arc_final.pt')