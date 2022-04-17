import numpy as np
import argparse

import torch
from models import seg_model
from utils import create_dir, viz_seg
from data_loader import get_eval_loader


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--rot', type=int, default=0, help='rotate')
    parser.add_argument('--vis_num', type=int, default=10, help='rotate')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--device', type=str, default="cuda:2")

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    # ind = np.random.choice(10000,args.num_points, replace=False)
    # test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    # test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    test_dataloader = get_eval_loader(args, 'seg')
    correct_point = 0
    num_point = 0
    pred_labels = []
    gt_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            point_clouds, labels = batch
            B = point_clouds.shape[0]
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)
            gt_labels.append(labels.data)
            
            labels = labels.reshape([-1])
            predictions = model(point_clouds)
            predictions = predictions.transpose(1,2)
            predictions = predictions.reshape([-1, args.num_seg_class])
            _, pred_label = torch.max(predictions.data, 1)

            correct_point += pred_label.eq(labels.data).cpu().sum().item()
            num_point += labels.view([-1,1]).size()[0]
            pred_labels.append(pred_label.reshape(B,-1))
            
    test_accuracy = correct_point / num_point
    pred_labels = torch.cat(pred_labels).detach().cpu()
    gt_labels = torch.cat(gt_labels).detach().cpu()

    # ------ TO DO: Make Prediction ------
    # test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))
    with open("output/seg/seg_result_np%i.txt"%(args.num_points), "w") as file:
        file.write(str(args)+ "\n")
        file.write("test_accuracy: %f \n"%test_accuracy)

    # Visualize Segmentation Result (Pred VS Ground Truth)
    test_data = test_dataloader.dataset.data
    # test_label = test_dataloader.dataset.label
    if args.vis_num > 0:
        random_vis = np.random.choice(int(test_data.shape[0]), args.vis_num, replace=False)
        for i in random_vis:
            vis_data = test_data[i,...]
            vis_label = gt_labels[i].detach().cpu()
            # vis_data = vis_data.reshape(-1,3)
            viz_seg(vis_data, vis_label, "{}/seg/gt_{}_nump_{}.gif".format(args.output_dir, i, args.num_points), args.device)
            viz_seg(vis_data, pred_labels[i], "{}/seg/pred_{}_nump_{}_.gif".format(args.output_dir, i, args.num_points), args.device)
        
    vis_data = test_data[args.i].unsqueeze(0).to(args.device)
    predict = model(vis_data)[0]
    _, pred_label = torch.max(predict.data, 0)
    viz_seg(test_data[args.i], gt_labels[args.i], "{}/seg/gt_{}_nump_{}_rot{}.gif".format(args.output_dir, args.exp_name, args.num_points, args.rot), args.device)
    viz_seg(test_data[args.i], pred_label, "{}/seg/pred_{}_nump_{}_rot{}.gif".format(args.output_dir, args.exp_name, args.num_points, args.rot), args.device)