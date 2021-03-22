from Data_read import read_data
import cv2
import numpy as np
def gray_image(root,plot_image):
    train_images_path,train_images_label,val_images_path,val_images_label=read_data(root,plot_image)
    train_images= []
    val_images =[]
    for image_path in train_images_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        train_images.append(image)
    for image_path in val_images_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        val_images.append(image)
    train_images=np.array(train_images)
    val_images=np.array(val_images)
    print(np.shape(train_images))
    print(np.shape(val_images))
    return train_images,train_images_label,val_images,val_images_label
if __name__ == '__main__':
    gray_image("data",False)

#       path -> image -> gray_image