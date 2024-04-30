import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torch


def resize_image(image, input_size):
    """
    Resize an image to the specified input size.

    Args:
    - image: PIL Image object
    - input_size: int, desired size for the longer edge of the image

    Returns:
    - image: resized PIL Image object
    """
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    return image


def format_results(result, filter=0):
    """
    Format model results into annotations.

    Args:
    - result: Ultralytics model result
    - filter: int, minimum area for valid annotation

    Returns:
    - annotations: list of formatted annotations
    """
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def box_prompt(masks, bbox):
    """
    Prompt model with bounding box and obtain corresponding mask.

    Args:
    - masks: tensor, binary masks
    - bbox: list, bounding box coordinates [xmin, ymin, xmax, ymax]

    Returns:
    - mask: numpy array, segmented mask
    - max_iou_index: int, index of the mask with highest IoU
    """
    h = masks.shape[1]
    w = masks.shape[2]
    
    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

    masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)

    return masks[max_iou_index].cpu().numpy(), max_iou_index


def point_prompt(masks, points, point_label):  
    """
    Prompt model with points and obtain corresponding mask.

    Args:
    - masks: list of dicts, masks data
    - points: list of lists, point coordinates [[x1, y1], [x2, y2], ...]
    - point_label: list of ints, point labels (1 for positive, 0 for negative)

    Returns:
    - mask: numpy array, segmented mask
    - index: int, index of the selected mask
    """
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]
    
    onemask = np.zeros((h, w))
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation['segmentation']
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1 and point_label[i] == 1:
                onemask[mask] = 1
            if mask[point[1], point[0]] == 1 and point_label[i] == 0:
                onemask[mask] = 0
    onemask = onemask >= 1
    return onemask, 0


def show_masks_on_image(image, masks):
    """
    Overlay masks on the input image.

    Args:
    - image: PIL Image object
    - masks: list of numpy arrays, segmented masks

    Returns:
    - image_with_mask: PIL Image object with masks overlayed
    """
    image_with_mask = image.convert("RGBA")
    
    for mask in masks:
        height, width = mask.shape
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]
        
        mask_array[mask, :] = color
        mask_image = Image.fromarray(mask_array)

        width, height = image_with_mask.size
        mask_image = mask_image.resize((width, height))
        
        image_with_mask = Image.alpha_composite(
            image_with_mask,
            mask_image)
    
    return image_with_mask


    

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_box(box, ax):
    """
    Display a bounding box on an image.

    Args:
        box (list): List containing the coordinates of the bounding box [x0, y0, x1, y1].
        ax (matplotlib.axes.Axes): Axes object to draw the bounding box on.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_boxes_on_image(raw_image, boxes):
    """
    Display multiple bounding boxes on an image.

    Args:
        raw_image (PIL.Image.Image): Raw input image.
        boxes (list): List containing bounding boxes in the format [[x0, y0, x1, y1], ...].
    """
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    """
    Display points on an image.

    Args:
        raw_image (PIL.Image.Image): Raw input image.
        input_points (list): List containing points coordinates in the format [[x1, y1], [x2, y2], ...].
        input_labels (list, optional): List containing labels for each point. Default is None.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points(coords, labels, ax, marker_size=375):
    """
    Display points on an axis.

    Args:
        coords (numpy.ndarray): Array containing points coordinates in the format [[x1, y1], [x2, y2], ...].
        labels (numpy.ndarray): Array containing labels for each point.
        ax (matplotlib.axes.Axes): Axes object to draw the points on.
        marker_size (int, optional): Size of the markers. Default is 375.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_masks_on_image(image, masks):
    """
    Overlay masks on an image.

    Args:
        image (PIL.Image.Image): Input image.
        masks (list): List containing binary masks.
    
    Returns:
        PIL.Image.Image: Image with overlaid masks.
    """
    image_with_mask = image.convert("RGBA")
    
    for mask in masks:
        height, width = mask.shape
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]
        
        mask_array[mask, :] = color
        mask_image = Image.fromarray(mask_array)

        width, height = image_with_mask.size
        mask_image = mask_image.resize((width, height))
        
        image_with_mask = Image.alpha_composite(image_with_mask, mask_image)
    
    return image_with_mask

def show_binary_mask(masks, scores):
    """
    Display binary masks with scores.

    Args:
        masks (torch.Tensor): Tensor containing binary masks.
        scores (torch.Tensor): Tensor containing scores.
    """
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    fig, ax = plt.subplots(figsize=(15, 15))
    idx = scores.tolist().index(max(scores))
    mask = masks[idx].cpu().detach()
    ax.imshow(np.array(masks[0,:,:]), cmap='gray')
    score = scores[idx]
    ax.title.set_text(f"Score: {score.item():.3f}")
    ax.axis("off")
    plt.show()
