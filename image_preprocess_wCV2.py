import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path

def preprocess(img_name):

    #Read image
    image=img_name+".jpg"
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    '''
    cv2.imshow("Grayscale image", img)
    k = cv2.waitKey(0)  #Wait for ESC keyboard stroke
    if(k==27):
        cv2.destroyAllWindows()
    '''
    #Denoising the image
    res_img =  cv2.fastNlMeansDenoising(img,None,10,7,21)
    ret,res_img = cv2.threshold(res_img,127,255,cv2.THRESH_TOZERO)
    '''
    cv2.imshow("Denoised Image in Grayscale", res_img)
    k = cv2.waitKey(0)  #Wait for ESC keyboard stroke
    if(k==27):
        cv2.destroyAllWindows()
    '''
    #Resizing the image
    res_img = cv2.resize(255-img, (28, 28), interpolation=cv2.INTER_AREA)
    '''
    cv2.imshow("Resized  grayscale image", res_img)
    k = cv2.waitKey(0)  #Wait for ESC keyboard stroke
    if(k==27):
        cv2.destroyAllWindows()
    '''
    #Converting to Binary image
    ret,res_img = cv2.threshold(res_img,127,255,cv2.THRESH_BINARY)
    '''
    cv2.imshow("Binary image", res_img)
    k = cv2.waitKey(0)  #Wait for ESC keyboard stroke
    if(k==27):
        cv2.destroyAllWindows()
    '''
    return res_img

'''
def main():
    name = input("Enter the name of the image: ")
    preprocess(name)

if __name__ == "__main__":
    main()
'''
