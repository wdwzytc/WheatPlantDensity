import cv2
import torch
import argparse
from PIL import Image
import numpy as np
import matplotlib as mpl

mpl.use('tkagg')
import matplotlib.pyplot as plt
from models import build_model

import torchvision.transforms as standard_transforms
import os


def wrap(image, wrap_thickness):
    dbg = True
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


def make_patch_borders_2(
        image,

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
    assert (h >= 256)
    assert (w >= 256)

    h_last = int(h - h // 768 * 768)
    if h_last == 0:
        # perfect
        h_lst = list(range(0, h, 768)) + [h]
    elif h_last >= 256:
        # ok for the last
        h_lst = list(range(0, h, 768)) + [h]
    elif h_last < 256:
        # the last space for h is not enough,
        # move forward
        h_lst = list(range(0, h, 768))
        h_lst[-1] -= (256 - h_last)
        h_lst += [h]
    else:
        raise ValueError("check image h")

    w_last = int(w - w // 768 * 768)
    if w_last == 0:
        # perfect
        w_lst = list(range(0, w, 768)) + [w]
    elif w_last >= 256:
        # ok for the last
        w_lst = list(range(0, w, 768)) + [w]
    elif w_last < 256:
        # the last space for h is not enough,
        # move forward
        w_lst = list(range(0, w, 768))
        w_lst[-1] -= (256 - w_last)
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


def get_p2p_model(weight_path='./weights/D20231111_best_mae.pth'):
    from run_test import get_args_parser
    parser = argparse.ArgumentParser(
        'P2PNet evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args([
        '--weight_path',
        weight_path,
    ])

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    model = build_model(args)
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model


def get_p2p_model_and_criterion(weight_path='./weights/D20231111_best_mae.pth'):
    from train import get_args_parser
    parser = argparse.ArgumentParser(
        'P2PNet evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args([])

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    model, criterion = build_model(args, training=True)

    if weight_path is not None:
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model, criterion


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


def partition_eval_merge(image, model, transform, device, wrap_thickness):
    # 1. wrap image to have a rounded size (128*n)
    # 1. crop into patches,
    # 2. model evaluate each patch,
    # 4. combine all results
    image_wrapped, h_offset_wr, w_offset_wr = wrap_rounding(image, size_factor=128)
    patch_borders = make_patch_borders_2(image_wrapped)
    points_eff_s = np.empty((0, 2), dtype=float)

    for patch_border in patch_borders:
        patch_wrapped = get_wrapped_patch(image_wrapped, patch_border,
                                          wrap_thickness=wrap_thickness)

        patch_transform = transform(patch_wrapped)
        samples = torch.Tensor(patch_transform).unsqueeze(0)
        samples = samples.to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        threshold = 0.5
        points = (outputs_points[outputs_scores > threshold]
                  .detach().cpu().numpy())

        del outputs
        del outputs_scores
        del outputs_points
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

    model = get_p2p_model()
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
    points = partition_eval_merge(image, model, transform, device, wrap_thickness=384)

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

        model = get_p2p_model()
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
        points = partition_eval_merge(image, model, transform, device, wrap_thickness=384)

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

        model = get_p2p_model()
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

        model = get_p2p_model()
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
    with Image.open(r"DATA_ROOT/temp/an.jpg") as fp:
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

    model = get_p2p_model()
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
    why_do_we_need_patch_wrap_thicker_than_320()
    torch.cuda.empty_cache()
    pass
