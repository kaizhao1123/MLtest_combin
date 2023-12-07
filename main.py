import sys
import os
import os.path
import tkinter as tk
import tracemalloc

from LDC import usingLDC, convertMaskToMat, convert01
from dexined import usingDexiNed, convertToMask
from dexined_multiple import usingDexiNed_multiple, combineImages


if __name__ == '__main__':

    window = tk.Tk()
    # some properties of the window.
    windowHeight = int(window.winfo_screenheight() / 1.5)
    windowWidth = int(window.winfo_screenwidth() / 2)


    def running():
        tracemalloc.start()

        # usingDexiNed()      # run model to get the contour of single image.
        # usingDexiNed_multiple()
        combineImages()

        # ********using LDC *********
        # usingLDC()
        # convertMaskToMat()
        # convert01()
        # ***************************


        # print("DIP time: --- %0.3f seconds ---" % (time() - startTime) + "\n")
        print("DIP memory: --- %0.2f M ---" % (tracemalloc.get_traced_memory()[0] / 1000000) + "\n")
        tracemalloc.stop()
        return


    button_run = tk.Button(window, text='Run', font=('Arial', 18), command=running)
    button_run.place(x=30, y=40, width=50, height=50)

    window.mainloop()


