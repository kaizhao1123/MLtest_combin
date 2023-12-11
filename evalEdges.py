from ctypes import *
import os
import cv2
import time
import numpy as np
from scipy.io import loadmat
from impl.toolbox import conv_tri, grad2
from impl.edges_eval_dir import edges_eval_dir
from impl.edges_eval_plot import edges_eval_plot
from argparse import ArgumentParser

# NOTE:
#    In NMS, `if edge < interp: out = 0`, I found that sometimes edge is very close to interp.
#    `edge = 10e-8` and `interp = 11e-8` in C, while `edge = 10e-8` and `interp = 9e-8` in python.
#    ** Such slight differences (11e-8 - 9e-8 = 2e-8) in precision **
#    ** would lead to very different results (`out = 0` in C and `out = edge` in python). **
#    Sadly, C implementation is not expected but needed :(
# C:/Users/Kai Zhao/PycharmProjects/edge_eval/
solver = cdll.LoadLibrary('./cxx/lib/solve_csa.so')
c_float_pointer = POINTER(c_float)
solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer, c_int, c_int, c_float, c_int, c_int]


def evalEdges():

    main_res_dir = 'results'
    data_name = 'BIPED'
    model = 'DXN'
    base_dir = os.path.join(main_res_dir, data_name + '-' + model)
    os.makedirs(base_dir, exist_ok=True)
    # At the end you will have in the base_dir, for example results/BIPED-DXN, the following folders>
    # * edge_maps= edges predicted from DXN model
    # * edge_nms=  Non-maximum-supression from edge_maps
    # * gt= GT of the dataset for evalution
    # * values = The computer results

    parser = ArgumentParser("edge eval")
    parser.add_argument("--alg", type=str, default=data_name + '-' + model, help="algorithms for plotting.")
    parser.add_argument("--model_name_list", type=str, default="hed", help="model name")
    parser.add_argument("--result_dir", type=str, default=base_dir, help="results directory")
    parser.add_argument("--file_format", type=str, default=".png", help=".mat or .npy, .png")
    parser.add_argument("--workers", type=int, default="-1", help="number workers, -1 for all workers")
    parser.add_argument("--version_detaile", type=str, default="20 epochs WD1e-8 ",
                        help="Additional details of the model")
    args = parser.parse_args()



    alg = [args.alg]  # algorithms for plotting
    model_name_list = [args.model_name_list]  # model name
    result_dir = os.path.abspath(args.result_dir)  # forward result directory
    save_dir = os.path.abspath(args.save_dir)  # nms result directory
    gt_dir = os.path.abspath(args.gt_dir)  # ground truth directory
    key = args.key  # x = scipy.io.loadmat(filename)[key]
    file_format = args.file_format  # ".mat" or ".npy"
    workers = args.workers  # number workers
    nms_process(model_name_list, result_dir, save_dir, key, file_format)
    eval_edge(alg, model_name_list, save_dir, gt_dir, workers)


#
def eval_edges(obj_path):
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]

    for model_name in model_name_list:
        tic = time.time()
        res_dir = os.path.join(result_dir, model_name)
        print(res_dir)
        edges_eval_dir(res_dir, gt_dir, thin=1, max_dist=0.0075, workers=workers)
        # edgesNmsMex(tmp_edge,O,1,5,1.03,8);
        toc = time.time()
        print("TIME: {}s".format(toc - tic))
        edges_eval_plot(res_dir, alg)


def nms_process_one_image(image, save_path=None, save=True):
    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(1.01), int(out.shape[1]), int(out.shape[0])
    solver.nms(out.ctypes.data_as(c_float_pointer),
               edge.ctypes.data_as(c_float_pointer),
               ori.ctypes.data_as(c_float_pointer),
               r, s, m, w, h)
    edge = np.round(out * 255).astype(np.uint8)
    if save:
        cv2.imwrite(save_path, edge)
    return edge


def nms_process(model_name_list, result_dir, save_dir, key=None, file_format=".mat"):
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]
    assert file_format in {".mat", ".npy"}
    assert os.path.isdir(result_dir)

    for model_name in model_name_list:
        model_save_dir = os.path.join(save_dir, model_name)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)

        for file in os.listdir(result_dir):
            save_name = os.path.join(model_save_dir, "{}.png".format(os.path.splitext(file)[0]))
            if os.path.isfile(save_name):
                continue

            if os.path.splitext(file)[-1] != file_format:
                continue
            abs_path = os.path.join(result_dir, file)
            if file_format == ".mat":
                assert key is not None
                image = loadmat(abs_path)[key]
            elif file_format == ".npy":
                image = np.load(abs_path)
            else:
                raise NotImplementedError
            nms_process_one_image(image, save_name, True)