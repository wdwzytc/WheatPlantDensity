from pathlib import Path
import cv2
import torch
import argparse
from PIL import Image
import numpy as np
import matplotlib as mpl

mpl.use('tkagg')
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
import os
from ultralytics import YOLO


def wrap(image, wrap_thickness):
    h, w, c = image.shape
    image_wrapped = np.zeros(
        (h + 2 * wrap_thickness, w + 2 * wrap_thickness, c),
        dtype=image.dtype,
    )
    image_wrapped[wrap_thickness:h + wrap_thickness, wrap_thickness:w + wrap_thickness, :] = image
    return image_wrapped


def wrap_rounding(image, size_factor: int):
    """
    Wrap an image to make its height and width be multiples of 128.
    This avoids shape changing in P2PNET
    """
    h, w, c = image.shape
    h_wrap = int((h // 128 + 1) * 128)
    w_wrap = int((w // 128 + 1) * 128)
    wrapped_image = np.zeros((h_wrap, w_wrap, c), dtype=np.uint8)
    h_start = (h_wrap - h) // 2
    h_end = h_start + h
    w_start = (w_wrap - w) // 2
    w_end = w_start + w
    wrapped_image[h_start:h_end, w_start:w_end] = image

    return wrapped_image, h_start, w_start


def make_patch_borders(
        image,
        minimum_patch_height=256,
        maximum_patch_height=768,
        minimum_patch_width=256,
        maximum_patch_width=768,

):
    r'''
    Given an image, calculate the borders
    to crop image into patches with no overlap
    and avoid small patches.

    minimum_patch_height=256,
    maximum_patch_height=768,
    minimum_patch_width=256,
    maximum_patch_width=768,

    patch_border = (h_start, h_end, w_start, w_end)

    '''
    h, w, c = image.shape

    assert (h % 128 == 0)
    assert (w % 128 == 0)
    assert (h >= minimum_patch_height)
    assert (w >= minimum_patch_width)

    h_last = int(h - h // maximum_patch_height * maximum_patch_height)
    if h_last == 0:
        # perfect
        h_lst = list(range(0, h, maximum_patch_height)) + [h]
    elif h_last >= minimum_patch_height:
        # ok for the last
        h_lst = list(range(0, h, maximum_patch_height)) + [h]
    elif h_last < minimum_patch_height:
        # the last space for h is not enough,
        # move forward
        h_lst = list(range(0, h, maximum_patch_height))
        h_lst[-1] -= (minimum_patch_height - h_last)
        h_lst += [h]
    else:
        raise ValueError("check image h")

    w_last = int(w - w // maximum_patch_width * maximum_patch_width)
    if w_last == 0:
        # perfect
        w_lst = list(range(0, w, maximum_patch_width)) + [w]
    elif w_last >= minimum_patch_width:
        # ok for the last
        w_lst = list(range(0, w, maximum_patch_width)) + [w]
    elif w_last < minimum_patch_width:
        # the last space for h is not enough,
        # move forward
        w_lst = list(range(0, w, maximum_patch_width))
        w_lst[-1] -= (minimum_patch_width - w_last)
        w_lst += [w]
    else:
        raise ValueError("check image w")

    patch_borders = []
    for hi in range(len(h_lst) - 1):
        for wi in range(len(w_lst) - 1):
            # (h_start, h_end, w_start, w_end)
            patch_borders.append(
                (h_lst[hi],
                 h_lst[hi + 1],
                 w_lst[wi],
                 w_lst[wi + 1])
            )

    return patch_borders


def get_wrapped_patch(
        image,
        patch_border,
        wrap_thickness,
):
    """
    Idea:
    Crop a patch from image.
    The patch has larger size than the patch_border.
    [h_start - wrap_thickness : h_end + wrap_thickness,
     w_start - wrap_thickness , w_end + wrap_thickness].
    Put value 0 to pixels out of the image.
    """

    # wrapping both the image and the (h_start, h_end, w_start, w_end)
    image_wrapped = wrap(image, wrap_thickness)

    h_start, h_end, w_start, w_end = patch_border
    h_start_wrapped = h_start
    w_start_wrapped = w_start
    h_end_wrapped = h_end + 2 * wrap_thickness
    w_end_wrapped = w_end + 2 * wrap_thickness

    patch = image_wrapped[
        h_start_wrapped:h_end_wrapped,
        w_start_wrapped:w_end_wrapped,
        :,]
    return patch


def get_yolo_model(weight_path=r'/home/tiancheng/PycharmProjects/YoloLeaftip/runs/detect/train13/weights/best.pt'):
    model = YOLO(weight_path)

    return model


def get_eff_proj_pts(points, image_shape, patch_border, wrap_thickness):
    """
    Get effective projected points of the original image

    (Ignore points in the wrap of the patch)
    """

    """
    The coordinate in "points" ([x, y])
    x (points[:,0]) is the distance from point to
      patch left-upper corner in the width direction
    y (points[:,1]) is the distance from point to
      patch left-upper corner in the height direction
    """

    h_img, w_img, c_img = image_shape

    h_patch_start, h_patch_end, w_patch_start, w_patch_end = patch_border
    h_patch = h_patch_end - h_patch_start
    w_patch = w_patch_end - w_patch_start

    # choose points inside the central part of patch
    chosen = np.logical_and.reduce((
        wrap_thickness <= points[:, 0],
        points[:, 0] <= wrap_thickness + w_patch,
        wrap_thickness <= points[:, 1],
        points[:, 1] <= wrap_thickness + h_patch
    ))
    effective_points = points[chosen, :]

    # remove patch-wrap from coordinates. = move to left-up.
    epp = effective_projected_points = effective_points - wrap_thickness

    # x, in the width direction
    epp[:, 0] = w_patch_start + epp[:, 0]

    # y, in the height direction
    epp[:, 1] = h_patch_start + epp[:, 1]

    return epp


def partition_eval_merge(image, model, wrap_thickness):
    # 1. wrap image to have a rounded size (128*n)
    # 1. crop into patches,
    # 2. model evaluate each patch,
    # 4. combine all results
    temp_out_path = '/home/tiancheng/PycharmProjects/YoloLeaftip/temp_data/temp.jpg'
    Path(temp_out_path).parent.mkdir(exist_ok=True, parents=True)
    image_wrapped, h_offset_wr, w_offset_wr = wrap_rounding(image, size_factor=128)
    patch_borders = make_patch_borders(image_wrapped, 256, 768, 256, 768)
    points_eff_s = np.empty((0, 2), dtype=float)

    for patch_border in patch_borders:
        patch_wrapped = get_wrapped_patch(image_wrapped, patch_border,
                                          wrap_thickness=wrap_thickness)
        if patch_wrapped.mean() < 1: continue  # skip black patches

        Image.fromarray(patch_wrapped).save(temp_out_path)

        image_long_side_size = max(patch_wrapped.shape[0],
                                   patch_wrapped.shape[1])
        outputs = model(temp_out_path, max_det=10000, verbose=False,
                        imgsz=image_long_side_size, )

        points = (outputs[0].boxes.xywh
                  .detach().cpu().numpy())[:, :2]

        del outputs
        torch.cuda.empty_cache()

        points_eff = get_eff_proj_pts(points, image_wrapped.shape, patch_border,
                                      wrap_thickness=wrap_thickness)
        points_eff_s = np.concatenate([points_eff_s, points_eff], axis=0)

        # test code:
        #   for patch and all detected points:
        #       plt.subplots()
        #       plt.imshow(patch_wrapped)
        #       plt.scatter([point[0] for point in points],
        #                   [point[1] for point in points])
        #   for image_wrapped and effective detected points:
        #       plt.subplots()
        #       plt.imshow(image_wrapped)
        #       plt.scatter([point[0] for point in points_eff_s],
        #                   [point[1] for point in points_eff_s])

    # fix offset caused by warp_rounding
    points_eff_s[:, 0] -= w_offset_wr
    points_eff_s[:, 1] -= h_offset_wr

    return points_eff_s.tolist()


if __name__ == '__main__':
    pass

    model = get_yolo_model(
        weight_path=r'/home/tiancheng/PycharmProjects/YoloLeaftip'
                    r'/runs/detect/train13/weights/best.pt'
    )

    p = r'/home/tiancheng/PycharmProjects/P2PNET_ROOT/DATA_ROOT/temp/example.jpg'

    result_out_path = (r'/home/tiancheng/PycharmProjects/P2PNET_ROOT/DATA_ROOT/temp/example_yolov8_result.json')

    with Image.open(str(p)) as fp:
        image = np.array(fp)
    points = partition_eval_merge(image, model, wrap_thickness=384)

    # visualize
    plt.subplots()
    plt.imshow(image)
    plt.scatter([point[0] for point in points],
                [point[1] for point in points],
                c='red',
                label='YOLOv8', )
    plt.legend()

    pass
