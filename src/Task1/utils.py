import cv2
import glob
import os


def read_image(image_name, type):
    return cv2.imread(image_name, type)


def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(file_name, image):
    cv2.imwrite(file_name + ".jpg", image)


'''
Example: print(len(read_images_from_folder("Train", "original_retinal_images", "jpg")))
'''


def read_images_from_folder(dir_path):
    image_list = []
    for image_file_name in glob.glob(dir_path):
        image = cv2.imread(image_file_name)
        if image is not None:
            image_list.append(image)

    return image_list

# Get the directory containing of the train and test dataset dynamically
def get_dir_path_with_train_test_dirs():
    project_dir = os.path.abspath('../..')
    if 'resources' in os.listdir(project_dir):
        return os.path.join(project_dir, 'resources', 'Task1')
    else:
        exit('Wrong check to the directory, please check your code!')


# Get the train folder
def get_train_dir():
    if 'Train' in os.listdir(get_dir_path_with_train_test_dirs()):
        return os.path.join(get_dir_path_with_train_test_dirs(), 'Train')
    else:
        exit('Check if the dataset is removed from resources')


# Get the test folder
def get_test_dir():
    if 'Test' in os.listdir(get_dir_path_with_train_test_dirs()):
        return os.path.join(get_dir_path_with_train_test_dirs(), 'Test')
    else:
        exit('Check if the dataset is removed from resources')
