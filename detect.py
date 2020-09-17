from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from yolo_models import *
from utils import *
from torch.autograd import Variable


DEVICE = torch.device("cpu")

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class SSDModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = torch.load(self.model_path, map_location=DEVICE)["model"]
        self.model.to(DEVICE)
        self.model.eval()

    def detect(self, image, min_score, max_overlap, top_k=5, suppress=None):
        """
        Detect objects in an image with a trained SSD300, and visualize the results.

        :param image: image, a PIL Image
        :param min_score: minimum threshold for a detected box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
        :return: annotated image, a PIL Image
        """

        # Transform
        image_tensor = normalize(to_tensor(resize(image)))

        # Move to default device
        image_tensor = image_tensor.to(DEVICE)

        # Forward prop.
        predicted_locs, predicted_scores = self.model(image_tensor.unsqueeze(0))
        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = self.model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                 max_overlap=max_overlap, top_k=top_k)
        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
        if det_labels == ['background']:
            # Just return original image
            return image

        # Annotate
        annotated_image = image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./arial.ttf", 15)
        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                det_labels[i]])
            # Text
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
        del draw
        return annotated_image, det_labels


class YOLOV3:
    def __init__(self, model_config_path, weights_path, class_names):
        self.model_config_path = model_config_path
        self.model_weights_path = weights_path
        self.model = Darknet(self.model_config_path).to(device)
        self.model.load_darknet_weights(self.model_weights_path)
        self.model.eval()
        self.classes = load_classes(class_names)

    def detect(self, image, min_score=0.6, nms_threshold=0.5):

        Tensor = torch.FloatTensor
        # image = image.resize((416, 416))
        input_imgs = Variable(transforms.ToTensor()(image).unsqueeze_(0).type(Tensor))
        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, min_score, nms_threshold)

        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections[0], image.size[:2])
            # Annotate
            annotated_image = image
            draw = ImageDraw.Draw(annotated_image)
            font = ImageFont.truetype("./arial.ttf", 15)
            det_labels = []
            for box_details in detections:
                box_location = box_details.tolist()[0:4]
                cls_pred = box_details.tolist()[-1]
                det_labels.append(self.classes[int(cls_pred)])
                draw.rectangle(xy=box_location, outline=coco_label_color_map[self.classes[int(cls_pred)]])
                draw.rectangle(xy=[l + 1. for l in box_location], outline=coco_label_color_map[self.classes[int(cls_pred)]])

                # Text
                text_size = font.getsize(self.classes[int(cls_pred)].upper())
                text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                textbox_location = [box_location[0], box_location[1] - text_size[1],
                                    box_location[0] + text_size[0] + 4.,
                                    box_location[1]]
                draw.rectangle(xy=textbox_location, fill=coco_label_color_map[self.classes[int(cls_pred)]])
                draw.text(xy=text_location, text=self.classes[int(cls_pred)].upper(), fill='white', font=font)
            del draw
            annotated_image = annotated_image.resize((416,416))
            return annotated_image, det_labels


if __name__ == '__main__':
    img_path = './img/test2.jpg'
    det = SSDModel('checkpoint_ssd300.pth.tar')
    image = Image.open(img_path, mode='r')
    image = image.convert('RGB')
    det.detect(image, min_score=0.8, max_overlap=0.5, top_k=200)
