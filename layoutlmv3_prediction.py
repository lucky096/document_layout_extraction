import numpy as np
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor #AutoProcessor # AutoModelForTokenClassification
from datasets import load_dataset 
from visualizations import get_label_id_mappings, draw_gt, unnormalize_box, draw_pred


if __name__ == "__main__":

    # this dataset uses the new Image feature :)
    dataset = load_dataset("nielsr/funsd-layoutlmv3")

    example = dataset["test"][0]
    print(example.keys())

    image = example["image"]
    width, height = image.size
    image = image.convert("RGB")
    boxes = example["bboxes"]
    labels = example["ner_tags"]
    
    id2label, label2id = get_label_id_mappings(dataset)
    print(id2label)
    
    # draw groundtruth
    # draw_gt(image, boxes, labels, id2label)

    # load model and processor
    LMV3_MODEL = "microsoft/layoutlmv3-base"
    model = LayoutLMv3ForTokenClassification.from_pretrained(LMV3_MODEL, id2label=id2label, label2id=label2id)
    processor = LayoutLMv3Processor.from_pretrained(LMV3_MODEL)

    # Inference using the labels
    # processor = LayoutLMv3Processor.from_pretrained(LMV3_MODEL, apply_ocr=False)

    # words = example["tokens"]
    # image = example["image"]
    # boxes = example["bboxes"]
    # word_labels = example["ner_tags"]

    # encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
    # for k,v in encoding.items():
    #     print(k,v.shape)

    # with torch.no_grad():
    #     outputs = model(**encoding)

    # logits = outputs.logits
    # print(logits.shape)

    # predictions = logits.argmax(dim=-1).squeeze().tolist()
    # print(predictions)

    # labels = encoding.labels.squeeze().tolist()
    # print(labels)

    # Inference without the labels
    encoding = processor(image, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop("offset_mapping")
    # print(encoding.keys())

    # forward pass
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()


    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    print(true_predictions)
    draw_pred(image, true_boxes, true_predictions)