import preprocessing as pre
import utils as utils
import argparse


def get_command_line_args():
    parser = argparse.ArgumentParser(description='Get info about the task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_category', '-m', default='Train',
                        help="Specify the model category like train / test")
    parser.add_argument('--directory', '-dir', metavar='INPUT',
                        help='Directory name of images', required=True)
    parser.add_argument('--image_format', '-imf', metavar='INPUT',
                        help='Image formatting type', required=True, default="jpg")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help="Learning rate of the model",
                        default=0.001)

    return parser.parse_args()


# Main function code start
args = get_command_line_args()

image = utils.read_image("../../resources/Task1/Train/original_retinal_images/IDRiD_02.jpg", 1)
print(image.shape)

images = utils.read_images_from_folder(args.model_category, args.directory, args.image_format)
print(len(images))

image = pre.scaleradius(image, 300)
image = pre.normalization(image)
print(image.shape)
image = pre.enhance(image)
utils.display_image(image)
