import torch

v = torch.Tensor([[1, 2, 1],
                  [4, 5, 6]])
r = torch.squeeze(v)
print(r.size())

t = torch.ones(20, 10, 2, 1)
r = torch.squeeze(t, 1)
print(r.size())


def get_command_line_args():
    parser = argparse.ArgumentParser(description='Get info about the task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_category', '-m', default='Train',
                        help="Specify the model category like train / test")
    parser.add_argument('--directory', '-dir', metavar='INPUT',
                        help='Directory name of images', required=False)
    # parser.add_argument('--image_format', '-imf', metavar='INPUT',
    # help='Image formatting type', required=True, default="jpg")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help="Learning rate of the model",
                        default=0.001)
    parser.add_argument('--batch_size', '-b', type=int,
                        help="Batch size of the model",
                        default=1)
    parser.add_argument('--no_epochs', '-noe', type=int,
                        help="No of epochs for the model",
                        default=1)
    parser.add_argument('--val_size', '-vs', type=int,
                        help="Validation Size",
                        default=0.2)

    return parser.parse_args()