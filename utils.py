import torchvision.ops as ops
import numpy as np
import random
import torch
import cv2
import os


def load_files():
    pillows_names = os.listdir('pillows/')
    pillows = [cv2.imread('pillows/' + pil, cv2.IMREAD_UNCHANGED) for pil in pillows_names]
    landscapes_names = os.listdir('images/')
    landscapes = [cv2.imread('images/' + img) for img in landscapes_names]

    return pillows, landscapes


def rotate_image(image, angle):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_matrix[0, 2] += bound_w / 2 - image_center[0]
    rotation_matrix[1, 2] += bound_h / 2 - image_center[1]

    rotated_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))

    return rotated_image


def generate_data(pils, images, img_id):
    pil_id = random.randint(0, len(pils)-1)
    image = images[img_id]
    pil = pils[pil_id]

    image = cv2.resize(image, (128, 128))
    pil = cv2.resize(pil, (32, 32))

    while True:
        random_angle = random.uniform(0, 360)
        rotated_image = rotate_image(pil, random_angle)

        max_x = image.shape[1] - rotated_image.shape[1]
        max_y = image.shape[0] - rotated_image.shape[0]
        if max_y > 0 and max_x > 0:
            break

    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)
    mask = rotated_image[:, :, 3]
    mask = cv2.merge((mask, mask, mask))

    rotated_image_rgb = rotated_image[:, :, :3]
    roi = image[random_y:random_y + rotated_image_rgb.shape[0],
                random_x:random_x + rotated_image_rgb.shape[1]]
    result = np.where(mask == (255, 255, 255), rotated_image_rgb, roi)

    image_cpy = image.copy()
    image_cpy[random_y:random_y + rotated_image_rgb.shape[0],
              random_x:random_x + rotated_image_rgb.shape[1]] = result

    img_mask = np.zeros_like(image_cpy)
    img_mask = img_mask[:, :, :1]
    mask = mask[:, :, :1]
    img_mask[random_y:random_y + rotated_image_rgb.shape[0],
             random_x:random_x + rotated_image_rgb.shape[1]] = mask

    x, y, w, h = cv2.boundingRect(img_mask)
    bbox = [x, y, x + w, y + h]
    # cv2.rectangle(image_cpy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
    # cv2.imshow('Bounding Boxes', image_cpy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image_cpy, bbox


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        bbox_cords = model(images)
        loss = criterion(bbox_cords, targets)
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()


def test(model, train_loader, device):
    model.eval()
    box_iou_list = []

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        box = model(images)
        print(targets.size)
        print(box.size)
        box_iou = ops.box_iou(targets, box.unsqueeze(0))
        box_iou_max, _ = torch.max(box_iou, dim=0)
        box_iou_list.append(float(box_iou_max))

    avg_iou = sum(box_iou_list) / len(train_loader)
    print(f"Average IOU: {avg_iou:.3f}")


if __name__ == "__main__":
    pills, imgs = load_files()
    generate_data(pills, imgs)

