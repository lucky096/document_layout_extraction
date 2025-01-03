from PIL import ImageDraw, ImageFont
from datasets.features import ClassLabel


label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}


def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    label = label[2:]
    if not label:
        return "other"
    return label

def get_label_id_mappings(dataset):
    features = dataset["train"].features
    label_column_name = "ner_tags"

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    else:
        label_list = get_label_list(dataset["train"][label_column_name])
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}
    
    return id2label, label2id


def draw_gt(image, boxes, labels, id2label):
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, label in zip(boxes, labels):
        actual_label = iob_to_label(id2label[label]).lower()
        box = unnormalize_box(box, width, height)
        draw.rectangle(box, outline=label2color[actual_label], width=2)
        draw.text((box[0] + 10, box[1] - 10), actual_label, fill=label2color[actual_label], font=font)

    image.show()

def draw_pred(image, boxes, predictions):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, prediction in zip(boxes, predictions):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label], width=2)
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

    image.show()