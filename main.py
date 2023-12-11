import sys
import os
import os.path
import tkinter as tk
import tracemalloc

from augment import augmentBoth
from cnn import usingDexiNed, usingLDC
from evalEdges import evalEdges
from image import combineImages


class Object(object):
    pass


if __name__ == '__main__':

    tracemalloc.start()     # trace the memory

    def app_path():
        """Returns the base application path."""
        if hasattr(sys, 'frozen'):
            # Handles PyInstaller
            return os.path.dirname(sys.executable)
        return os.path.dirname(__file__)

    # create paths
    path_root = app_path() + '/'
    print("root path: ", path_root)

    path_sample_input = path_root + 'sample_pic/'
    path_sample_output = path_root + 'sample_result/'
    path_test_input = path_root + 'test_pic/'
    path_test_output = path_root + 'test_result/'
    path_aug_input = path_root + 'augment_input/'
    path_aug_output = path_root + 'augment_output/'

    if not os.path.exists(path_sample_input):
        os.mkdir(path_sample_input)
    if not os.path.exists(path_sample_output):
        os.mkdir(path_sample_output)
    if not os.path.exists(path_test_input):
        os.mkdir(path_test_input)
    if not os.path.exists(path_test_output):
        os.mkdir(path_test_output)
    if not os.path.exists(path_aug_input):
        os.mkdir(path_aug_input)
    if not os.path.exists(path_aug_output):
        os.mkdir(path_aug_output)
    ###########################################

    # create object for paths
    obj_path_sample = Object()
    obj_path_sample.path_root = path_root
    obj_path_sample.path_pic = path_sample_input
    obj_path_sample.path_result = path_sample_output

    obj_path_test = Object()
    obj_path_test.path_root = path_root
    obj_path_test.path_pic = path_test_input
    obj_path_test.path_result = path_test_output

    obj_path_aug = Object()
    obj_path_aug.path_root = path_root
    obj_path_aug.path_pic = path_aug_input
    obj_path_aug.path_result = path_aug_output

    # ******** using DexiNed *********************************************
    # usingDexiNed(obj_path_sample)  # generate the contour images from the input images using the model.
    # combineImages(obj_path_sample, 'dexined')  # remove the bottom noise of the contour images, and save to mask images.
    # ***************************

    # ********using LDC ***************************************************
    # usingLDC(obj_path_sample)
    # combineImages(obj_path_sample, 'ldc')
    # convertMaskToMat()
    # convert01()
    # ***************************

    # ******** augment images ***************************************************
    # augmentBoth(obj_path_aug)
    # ***************************

    # ******** eval edges ***************************************************
    evalEdges()
    # ***************************

    # print("DIP time: --- %0.3f seconds ---" % (time() - startTime) + "\n")
    print("DIP memory: --- %0.2f M ---" % (tracemalloc.get_traced_memory()[0] / 1000000) + "\n")
    tracemalloc.stop()



    # create a UI

    # window = tk.Tk()
    # # some properties of the window.
    # windowHeight = int(window.winfo_screenheight() / 1.5)
    # windowWidth = int(window.winfo_screenwidth() / 2)
    #
    #
    # def running():
    #     tracemalloc.start()
    #
    #     print("DIP memory: --- %0.2f M ---" % (tracemalloc.get_traced_memory()[0] / 1000000) + "\n")
    #     tracemalloc.stop()
    #     return
    #
    # button_run = tk.Button(window, text='Run', font=('Arial', 18), command=running)
    # button_run.place(x=30, y=40, width=50, height=50)
    #
    # window.mainloop()


