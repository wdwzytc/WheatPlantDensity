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
import json
from tqdm import tqdm


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
            :, ]
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


def main_example():
    """
    code example:
        points = main_example()
        d = {'image_name.jpg': {
            'predicted points': points,
            'manually labeled points': None,
        }}
        with open('temp.json', 'w') as fp:
            json.dump(d, fp, indent=4)
    """
    # 0. prepare directory, image, model and transformation
    # 1. crop image into patches,
    # 2. model evaluate each patch,
    # 3. combine different evaluations together.

    # 0. prepare directory, image, model and transformation

    # image

    with Image.open(
            r'/home/tiancheng/PycharmProjects'
            r'/P2PNET_ROOT/0_archive_working_folders_before'
            r'/M001 Avignon23_Fourque22_SdG22/01238.jpg') as fp:
        image = np.array(fp)

    # with Image.open(r'DATA_ROOT/independent_dataset'
    #                 r'/angle45_Gardanne_2022_01_03_'
    #                 r'plot1_densityTreatment0_33_soft'
    #                 r'_wheat_DSC09268_0.jpg') as fp:
    #     image = np.array(fp)

    model = get_yolo_model()

    # wrap, patchify, evaluate, unpatchify, unwrap.
    points = partition_eval_merge(image, model, wrap_thickness=384)

    # visualize
    plt.subplots()
    plt.imshow(image)
    plt.scatter([point[0] for point in points],
                [point[1] for point in points])

    return points


def compare_result_patch_vs_whole():
    """
    2023-10-25, there are slight differences. why?

    idea 1: check which one is right. right = take different sizes of patches that is large enough, with targeted point patch as the center, and check the outputs.
    idea 2: check with the last patch.

    answer 1: wraps thicker than 320 may solve this.
        see function
        why_do_we_need_patch_wrap_thicker_than_320()
    """

    def main_feed_patches():
        # 0. prepare directory, image, model and transformation
        # 1. crop image into patches,
        # 2. model evaluate each patch,
        # 3. combine different evaluations together.

        # 0. prepare directory, image, model and transformation
        p2pnet_root = r'/home/tiancheng/PycharmProjects/P2PNET_ROOT'
        os.chdir(p2pnet_root)

        # image

        with Image.open(r"DATA_ROOT/temp/an2.jpg") as fp:
            image = np.array(fp)

        # with Image.open(r'DATA_ROOT/independent_dataset'
        #                 r'/angle45_Gardanne_2022_01_03_'
        #                 r'plot1_densityTreatment0_33_soft'
        #                 r'_wheat_DSC09268_0.jpg') as fp:
        #     image = np.array(fp)

        model = get_yolo_model()
        device = torch.device('cuda')
        model.to(device)
        model.eval()

        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

        # wrap, patchify, evaluate, unpatchify, unwrap.
        points = partition_eval_merge(image, model, wrap_thickness=384)

        # visualize
        plt.subplots()
        plt.imshow(image)
        plt.scatter([point[0] for point in points],
                    [point[1] for point in points])

        return points

    def main_feed_a_whole_image():
        p2pnet_root = r'/home/tiancheng/PycharmProjects/P2PNET_ROOT'
        os.chdir(p2pnet_root)

        with Image.open(r"DATA_ROOT/temp/an2.jpg") as fp:
            image = np.array(fp)

        model = get_yolo_model()
        device = torch.device('cuda')
        model.to(device)
        model.eval()

        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

        image_wrap, h_offset_wr, w_offset_wr = wrap_rounding(image, 128)
        image_wrap = wrap(image_wrap, wrap_thickness=384)
        h_offset_wr += 384
        w_offset_wr += 384

        img_t = transform(image_wrap)
        samples = torch.Tensor(img_t).unsqueeze(0)
        samples = samples.to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        threshold = 0.5
        points = (outputs_points[outputs_scores > threshold]
                  .detach().cpu().numpy())
        points[:, 0] -= w_offset_wr
        points[:, 1] -= h_offset_wr
        points = points.tolist()
        # visualize
        plt.subplots()
        plt.imshow(image)
        plt.scatter([point[0] for point in points],
                    [point[1] for point in points])

        return points

    pp = points_with_patchify = np.array(main_feed_patches()).astype(np.float32)
    pw = points_as_a_whole = np.array(main_feed_a_whole_image()).astype(np.float32)

    pp = pp[np.argsort(pp[:, 0]), :]
    pw = pw[np.argsort(pw[:, 0]), :]

    ind_discor = np.apply_along_axis(np.all, 1, ~np.isclose(pp, pw))
    print("where they did not match:")
    print(pp[ind_discor])
    print(pw[ind_discor])

    return


def why_do_we_need_patch_wrap_thicker_than_320():
    """
    by comparing the results across different patch_sizes,
    the patch_size of 640*640 was found to be the size,
    where the DL output gets stable. Thus, we will need
    patch-wrap of 320 thickness.
    """

    def model_eval_patch(image, patch_center, patch_side_length):
        # patch_side_length = 512
        t = patch_thickness = patch_side_length // 2
        hp, wp = patch_center

        image_wrapped = np.zeros(
            (h + 2 * t,
             w + 2 * t,
             c),
            dtype=np.uint8,
        )
        image_wrapped[t:h + t, t:w + t, :] = image
        patch = image_wrapped[hp:hp + 2 * t, wp:wp + 2 * t, :]

        model = get_yolo_model()
        device = torch.device('cuda')
        model.to(device)
        model.eval()
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

        patch_t = transform(patch)
        samples = torch.Tensor(patch_t).unsqueeze(0)
        samples = samples.to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        threshold = 0.5
        points = (outputs_points[outputs_scores > threshold]
                  .detach().cpu().numpy())
        plt.subplots()
        plt.suptitle(f'patch: {patch_side_length} * {patch_side_length}')
        plt.imshow(patch)
        plt.scatter(points[:, 0], points[:, 1])
        print("corrdinates of points, from the center of patch:")
        print(f"patch_size: {patch_side_length} * {patch_side_length}")
        print(points - t)

    p2pnet_root = r'/home/tiancheng/PycharmProjects/P2PNET_ROOT'
    os.chdir(p2pnet_root)
    with Image.open(
            r"C:\Users\Yang\PycharmProjects\2024-3-27_Register_Avignon_Images_for_P2PNET\data_root\x_imgs_for_ubuntu\01238.jpg") as fp:
        image = np.array(fp)
        h, w, c = image.shape
    patch_center = hp, wp = 800, 1600

    for patch_side_length in [256, 384, 512, 640, 768, 896, 1024]:
        model_eval_patch(image, patch_center, patch_side_length)
        torch.cuda.empty_cache()

    print("""
    # Then, by comparing the results across different patch_sizes,
    # the patch_size of 640*640 was found to be the size,
    # where the DL output of the center part gets stable.
    # Thus, we will need patch-wrap of 320 thickness.
    """)


def d20231101_run_p2pnet_eval():
    p2pnet_root = r'/home/tiancheng/PycharmProjects/P2PNET_ROOT'
    os.chdir(p2pnet_root)

    with Image.open(r"DATA_ROOT/temp/an2.jpg") as fp:
        image = np.array(fp)

    model = get_yolo_model()
    device = torch.device('cuda')
    model.to(device)
    model.eval()

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    image_wrap, h_offset_wr, w_offset_wr = wrap_rounding(image, 128)
    img_t = transform(image_wrap)
    samples = torch.Tensor(img_t).unsqueeze(0)
    samples = samples.to(device)
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(
        outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    threshold = 0.5
    points = (outputs_points[outputs_scores > threshold]
              .detach().cpu().numpy())
    points[:, 0] -= w_offset_wr
    points[:, 1] -= h_offset_wr
    points = points.tolist()
    # visualize
    plt.subplots()
    plt.imshow(image)
    plt.scatter([point[0] for point in points],
                [point[1] for point in points])


if __name__ == '__main__':
    # main
    # run partition_eval_merge on a folder of images
    input_folder = (
        r'/home/tiancheng/PycharmProjects'
        r'/P2PNET_ROOT'
        r'/0_archive_working_folders_before'
        r'/M001 Avignon23_Fourque22_SdG22')
    model = get_yolo_model(
        weight_path=r'/home/tiancheng/PycharmProjects/YoloLeaftip'
                    r'/runs/detect/train13/weights/best.pt'
    )
    img_p_lst = [p for p in Path(input_folder).iterdir()
                 if p.suffix.lower() in ['.jpg', '.png']]
    result_out_path = (r'/home/tiancheng/PycharmProjects'
                       r'/YoloLeaftip'
                       r'/temp_YOLOv8_detect_results'
                       r'_patchification_as_P2PNET.json')

    r = yolo_detect_result_dict = {}
    for p in tqdm(img_p_lst):
        if p.suffix.lower() not in ['.jpg', '.png']: continue

        with Image.open(str(p)) as fp:
            image = np.array(fp)
        points = partition_eval_merge(image, model, wrap_thickness=384)
        # collect outputs as dictionary: {stem: lst of points, ...}
        r[p.name] = points
        dbg = True

    with open(result_out_path, 'w') as fp:
        json.dump(r, fp)

    pass
