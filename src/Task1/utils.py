import cv2
import glob


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
def read_images_from_folder(model_category, directory_name, extension_type):
    image_list = []
    dir_path = "../../resources/Task1/" + model_category + "/" + directory_name + "/*." + extension_type
    for image_file_name in glob.glob(dir_path):
        image = cv2.imread(image_file_name)
        if image is not None:
            image_list.append(image)

    return image_list

