#
# Codes from P2PNet (https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
# P2PNet used codes from DETR (https://github.com/facebookresearch/detr).
#   Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Sylvain and Tiancheng modified the codes. 2025-4-17 17:14:16.
############################################################
import os
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as standard_transforms
from scipy.optimize import linear_sum_assignment


class NestedTensor(object):
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    # # Code from P2PNet (https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet),
    # # # .util.misc
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list):
        out = []

        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)

        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    """ResNet backbone with frozen BatchNorm."""

    @staticmethod
    def make_layers(cfg, batch_norm=False, sync=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    if sync:
                        print('use sync backbone')
                        layers += [conv2d, nn.SyncBatchNorm(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    def vgg16_bn(pretrained=False, sync=False, vgg16bn_weight_path=None, **kwargs):
        r"""VGG 16-layer model (configuration "D") with batch normalization
        `"Very Deep Convolutional Networks For Large-Scale Image Recognition"
        <https://arxiv.org/pdf/1409.1556.pdf>`_

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """

        batch_norm = True
        cfgs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
        }
        cfg = 'D'
        model_path_vgg16bn = vgg16bn_weight_path

        if pretrained:
            kwargs['init_weights'] = False

        model = VGG(Backbone_VGG.make_layers(cfgs[cfg], batch_norm=batch_norm, sync=sync), **kwargs)

        if pretrained:
            state_dict = torch.load(model_path_vgg16bn)
            model.load_state_dict(state_dict)

        return model

    def __init__(self, name: str, return_interm_layers: bool, vgg16bn_weight_path: str):
        if name == 'vgg16_bn':
            backbone = Backbone_VGG.vgg16_bn(vgg16bn_weight_path=vgg16bn_weight_path, pretrained=True)
        else:
            raise ValueError('only support vgg16_bn here')
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


class RegressionModel(nn.Module):
    "P2PNet, regression branch"

    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    "P2PNet, classification branch"

    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    @staticmethod
    def _shift(shape, stride, anchor_points):
        'shift the meta-anchor to get an anchor points'
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchor_points.shape[0]
        K = shifts.shape[0]
        all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
        all_anchor_points = all_anchor_points.reshape((K * A, 2))

        return all_anchor_points

    @staticmethod
    def _generate_anchor_points(stride=16, row=3, line=3):
        'generate the reference points in grid layout'
        row_step = stride / row
        line_step = stride / line

        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        anchor_points = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        return anchor_points

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = self._generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted_anchor_points = self._shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))


class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = torch.nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = torch.nn.functional.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        num_boxes = num_points.item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses


class HungarianMatcher_Crowd(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and points
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["point"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L2 cost between point
        cost_point = torch.cdist(out_points, tgt_points, p=2)

        # Compute the giou cost between point

        # Final cost matrix
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["point"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256,
                                                  num_classes=self.num_classes,
                                                  num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)

        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[1]) * 100  # 8x
        classification = self.classification(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord}

        return out

    # todo: put these in "inference.py" (or class Inference?).
    #   reason: As recommended by ChatGPT,
    #   "P2PNet.py" should only define
    #   the model and forward() function.
    #   So, the functions below should not be here.
    @staticmethod
    def get_detected_points(p2pnet_model, image):
        device = torch.device('cuda')
        p2pnet_model.to(device)
        p2pnet_model.eval()
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        image_transform = transform(image)
        samples = torch.Tensor(image_transform).unsqueeze(0)
        samples = samples.to(device)
        outputs = p2pnet_model(samples)
        outputs_scores = torch.nn.functional.softmax(
            outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        threshold = 0.5  # default value in P2PNet
        points = (outputs_points[outputs_scores > threshold]
                  .detach().cpu().numpy().tolist())
        del outputs
        del outputs_scores
        del outputs_points
        torch.cuda.empty_cache()
        return points

    @staticmethod
    def get_detected_points_using_patchification(p2pnet_model, image,
                                                 wrap_thickness=384):
        r"""# 1. wrap image to have a rounded size (128*n, n is an integer)
        # 2. crop into patches,
        # 3. model evaluate each patch,
        # 4. combine all results"""

        def wrap(image, wrap_thickness):
            h, w, c = image.shape
            image_wrapped = np.zeros(
                (h + 2 * wrap_thickness, w + 2 * wrap_thickness, c),
                dtype=image.dtype,
            )
            image_wrapped[wrap_thickness:h + wrap_thickness, wrap_thickness:w + wrap_thickness, :] = image
            return image_wrapped

        def wrap_rounding(image):
            """
            Wrap an image to make its height and width be multiples of 128.
            This avoids changing shapes in P2PNET
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
                :,]
            return patch

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

            if not points:  # for empty list
                return np.empty((0, 2))

            points = np.array(points)
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

        image_wrapped, h_offset_wr, w_offset_wr = wrap_rounding(image)
        patch_borders = make_patch_borders_2(image_wrapped)
        points_eff_s = np.empty((0, 2), dtype=float)

        for patch_border in patch_borders:
            patch_wrapped = get_wrapped_patch(image_wrapped, patch_border,
                                              wrap_thickness=wrap_thickness)
            points = P2PNet.get_detected_points(p2pnet_model, patch_wrapped)

            points_eff = get_eff_proj_pts(points, image_wrapped.shape, patch_border,
                                          wrap_thickness=wrap_thickness)
            points_eff_s = np.concatenate([points_eff_s, points_eff], axis=0)

        # fix offset caused by warp_rounding
        points_eff_s[:, 0] -= w_offset_wr
        points_eff_s[:, 1] -= h_offset_wr

        return points_eff_s

    @staticmethod
    def get_p2p_model(p2pnet_weight_path=None,
                      vgg16bn_weight_path=None,
                      backbone='vgg16_bn',
                      row=2,
                      line=2,
                      gpu_id=0, ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        backbone = Backbone_VGG(backbone, True, vgg16bn_weight_path)
        model = P2PNet(backbone, row, line)

        if p2pnet_weight_path is not None:
            checkpoint = torch.load(p2pnet_weight_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

        return model

    @staticmethod
    def get_criterion(num_classes=1,
                      loss_ce=1,
                      loss_points=0.0002,
                      cost_class=1,
                      cost_point=0.05,
                      eos_coef=0.5, ):
        weight_dict = {'loss_ce': loss_ce, 'loss_points': loss_points}
        losses = ['labels', 'points']
        matcher = HungarianMatcher_Crowd(
            cost_class=cost_class,
            cost_point=cost_point)
        criterion = SetCriterion_Crowd(
            num_classes,
            matcher=matcher, weight_dict=weight_dict,
            eos_coef=eos_coef, losses=losses)
        return criterion


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path

    p2pnet_model = P2PNet.get_p2p_model(
        p2pnet_weight_path='./data/weights/N036_LeafTip_BestMae.pth',
        vgg16bn_weight_path='./data/weights/vgg16_bn-6c64b313.pth',
    )

    test_image_paths = Path(r'./data/images').rglob('*.jpg')
    for test_image_path in test_image_paths:
        with Image.open(test_image_path) as fp:
            rgb = np.array(fp)
        points = P2PNet.get_detected_points_using_patchification(p2pnet_model, rgb)
        fig, ax = plt.subplots()
        points_arr = np.array(points)
        ax.imshow(rgb)
        ax.scatter(points_arr[:, 0], points_arr[:, 1])
        ax.set_title('P2PNet Model Prediction')

    plt.pause(10)
