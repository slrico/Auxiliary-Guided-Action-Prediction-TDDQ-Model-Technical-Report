import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from transformers import CLIPProcessor, CLIPTokenizer
from torchvision import transforms
from PIL import Image
import torch
import spacy
import io


nlp = spacy.load("en_core_web_sm")

def parse_question_semantics(question_text):
    doc = nlp(question_text)
    entities = [ent.text for ent in doc.ents]
    actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    return {
        "entities": entities,
        "actions": actions,
        "raw": question_text
    }
    

def preprocess_example(example, label_to_prototype, tokenizer, image_transform):
    hazard_label = example['hazard']
    normalized_label = hazard_label.strip().lower()
    hazard_prototype = label_to_prototype.get(normalized_label, "unknown")
    image_data = example['image']
    try:
        if hasattr(image_data, 'convert'):
            img = image_data.convert("RGB")
        elif isinstance(image_data, str):
            import requests
            if image_data.startswith("http"):
                response = requests.get(image_data)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(image_data).convert("RGB")
        elif isinstance(image_data, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif isinstance(image_data, dict):
            if 'bytes' in image_data:
                img = Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
            elif 'path' in image_data:
                img = Image.open(image_data['path']).convert("RGB")
            else:
                raise ValueError(f"Unexpected image dict format: {image_data.keys()}")
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
    except Exception as e:
        print(f"⚠️ Error loading image for question_id {example.get('question_id', 'unknown')}: {e}")
        img = Image.new("RGB", (224, 224))  # fallback blank image

    img_tensor = image_transform(img)

   
    question_text = example['question'].strip()
    question_tokens = tokenizer(
        question_text,
        padding='max_length',
        max_length=32,
        truncation=True,
        return_tensors='pt'
    )
    question_semantics = parse_question_semantics(question_text)


    raw_bbox = example['bounding_box']
    bbox_list = parse_bounding_box(raw_bbox)
    bbox_norm = safe_bbox_norm(bbox_list)
    speed_norm = safe_speed_norm(example['plausible_speed'])


    return {
        "image": img_tensor,
        "question_input_ids": question_tokens['input_ids'].squeeze(0),
        "question_attention_mask": question_tokens['attention_mask'].squeeze(0),
        "bounding_box_raw": raw_bbox,
        "bounding_box_parsed": bbox_list,
        "bounding_box": bbox_norm,
        "plausible_speed": speed_norm,
        "hazard_label": hazard_label,
        "hazard_prototype": hazard_prototype,
        "question": question_text,
        "question_semantics": question_semantics,
        "question_id": example['question_id']
    }
    
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

processed_dataset = []
for example in dataset["train"]:
    processed = preprocess_example(example, label_to_prototype, tokenizer, image_transform)
    processed_dataset.append(processed) 
    
for i, item in enumerate(processed_dataset[:3]):
    print(f"\n🔎 Example {i + 1}")
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
unknowns = [ex for ex in processed_dataset if ex['hazard_prototype'] == "unknown"]
print(f"⚠️ Found {len(unknowns)} examples with unknown hazard prototype.")
empty_semantics = [ex for ex in processed_dataset if not ex['question_semantics']['actions']]
print(f"🧠 Found {len(empty_semantics)} questions with no detected actions.")

sample = processed_dataset[0]
img = TF.to_pil_image(sample['image'])
plt.imshow(img)
plt.title(sample['question'])
plt.axis('off')
plt.show()
