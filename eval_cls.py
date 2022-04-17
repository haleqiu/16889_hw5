import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, vis_cls
from data_loader import get_eval_loader

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--rot', type=int, default=0, help='rotate')
    parser.add_argument('--vis_num', type=int, default=10, help='rotate')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    # ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    # test_label = torch.from_numpy(np.load(args.test_label))
    test_dataloader = get_eval_loader(args)
    correct_obj = 0
    num_obj = 0
    pred_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)
            
            outputs = model(point_clouds)
            _, pred_label = torch.max(outputs.data, 1)

            pred_labels.append(pred_label)
            correct_obj += pred_label.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

    test_accuracy = correct_obj / num_obj

    pred_labels = torch.cat(pred_labels).detach().cpu()

    # ------ TO DO: Make Prediction -----
    test_data = test_dataloader.dataset.data
    test_label = test_dataloader.dataset.label
    if args.vis_num > 0:
        random_vis = np.random.choice(int(test_data.shape[0]), 10, replace=False)
        for i in random_vis:
            vis_data = test_data[i,...]
            vis_label = test_label[i].detach().cpu()
            # vis_data = vis_data.reshape(-1,3)
            vis_cls(vis_data, "%s/cls/cls_%i_label_%i_nump_%i.gif"%(args.output_dir, i, vis_label, args.num_points), args.device)

        # visualize false samples
        false_mask = pred_labels != test_label
        false_index =torch.nonzero(false_mask).numpy()
        for fidx in false_index:
            _data = test_data[fidx[0], ...]
            _pred = pred_labels[fidx].detach().cpu()
            _vis = test_label[fidx].detach().cpu()
            vis_cls(_data, "%s/cls/cls%i_false%i_label%i_nump%i.gif"%(args.output_dir, fidx, _pred, _vis, args.num_points), args.device)

    vis_data = test_data[args.i,...]
    vis_label = pred_labels[args.i].detach().cpu()
    # vis_data = vis_data.reshape(-1,3)
    vis_cls(vis_data, "%s/cls/exp_cls_%i_label_%i_nump_%i_rot%i.gif"%(args.output_dir, args.i, vis_label, args.num_points, args.rot), args.device)

    # Compute Accuracy
    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))
    with open("%s/cls/cls_result_np%i_rot%i.txt"%(args.output_dir, args.num_points, args.rot), "w") as file:
        file.write(str(args)+ "\n")
        file.write("test_accuracy: %f \n"%test_accuracy)
