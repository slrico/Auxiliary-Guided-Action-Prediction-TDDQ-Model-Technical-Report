import numpy as np
import ast
from collections import Counter
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


dataset = load_dataset("DHPR/Driving-Hazard-Prediction-and-Reasoning", streaming=True)

hazard_counts = Counter(
    ex['hazard'].strip()
    for ex in dataset['train']
    if ex['hazard'].strip()
)
label_set = list(hazard_counts.keys())


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

label_set = list(set(ex['hazard'].strip() for ex in dataset['train'] if ex['hazard'].strip()))
label_set = [label for label in label_set if label]  
label_embeddings = embedding_model.encode(label_set, normalize_embeddings=True)

print("✅ Embedded hazard labels:", len(label_set))
print("🔤 Sample labels:", label_set[:5])


hazard_counts = Counter(ex['hazard'].strip() for ex in dataset['train'] if ex['hazard'].strip())
prototypes = [label for label, _ in hazard_counts.most_common(8)]
prototype_embeddings = embedding_model.encode(prototypes, normalize_embeddings=True)

print("📌 Selected prototypes:", prototypes)


#### Group Labels by Prototypes - (Semantic Clustering)
similarities = cosine_similarity(label_embeddings, prototype_embeddings)
best_indices = np.argmax(similarities, axis=1)
prototype_groups = {proto: [] for proto in prototypes}
for label, proto_idx in zip(label_set, best_indices):
    proto = prototypes[proto_idx]
    prototype_groups[proto].append(label)

print("🧩 Grouped labels by prototype:")
for proto, group in prototype_groups.items():
    print(f"  {proto} → {len(group)} labels")
    
#### Build Labels -> Prototypes Mapping
def normalize(text):
    return text.strip().lower()

label_to_prototype = {
    normalize(label): proto
    for proto, group in prototype_groups.items()
    for label in group
}

print("🔗 Sample label-to-prototype mapping:")
for label in list(label_to_prototype.keys())[:5]:
    print(f"  {label} → {label_to_prototype[label]}")
    
def assign_prototype(example, label_to_prototype):
    hazard = normalize(example['hazard'])
    return label_to_prototype.get(hazard, "unknown")

for example in dataset["train"]:
    hazard_label = example['hazard']
    hazard_category = label_to_prototype.get(normalize(hazard_label), "unknown")
    print(f"  Hazard: {hazard_label} → Prototype Group: {hazard_category}")

##### Group Prototypes Similarities 
for proto, group in prototype_groups.items():
    print(f"\n🔍 Prototype: {proto}")
    group_embeddings = embedding_model.encode(group, normalize_embeddings=True)
    sim_matrix = cosine_similarity(group_embeddings)
    avg_sim = np.mean(sim_matrix[np.triu_indices(len(group), k=1)])
    print(f"  Avg intra-group similarity: {avg_sim:.3f}")
    
def safe_bbox_norm(bbox, img_size=224):
    norm_bbox = []
    for coord in bbox:
        try:
            coord_f = float(coord)
        except (ValueError, TypeError):
            coord_f = 0.0  
        norm_bbox.append(coord_f / img_size)
    return norm_bbox

def safe_speed_norm(speed, max_speed=100):
    try:
        speed_f = float(speed)
    except (ValueError, TypeError):
        speed_f = 0.0
    return speed_f / max_speed


def parse_bounding_box(raw_bbox):
    if isinstance(raw_bbox, list):
        if len(raw_bbox) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        first_bbox = raw_bbox[0]
        if isinstance(first_bbox, (list, tuple)):
            try:
                return [float(c) for c in first_bbox]
            except Exception as e:
                print("Error converting bounding box elements to float:", e)
                return [0.0, 0.0, 0.0, 0.0]
        else:
            try:
                return [float(first_bbox)] * 4
            except Exception as e:
                print("Error converting bounding box to floats:", e)
                return [0.0, 0.0, 0.0, 0.0]
    elif isinstance(raw_bbox, str):
        try:
            parsed = ast.literal_eval(raw_bbox)
            if isinstance(parsed, (list, tuple)):
                # In case parsed is list of lists, apply same logic
                if len(parsed) == 0:
                    return [0.0, 0.0, 0.0, 0.0]
                first_bbox = parsed[0] if isinstance(parsed[0], (list, tuple)) else parsed
                return [float(c) for c in first_bbox]
            else:
                print(f"Parsed bounding box is not a list/tuple: {parsed}")
                return [0.0, 0.0, 0.0, 0.0]
        except Exception as e:
            print("Error parsing bounding_box string:", e)
            return [0.0, 0.0, 0.0, 0.0]
    else:
        print(f"Unexpected bounding_box type: {type(raw_bbox)}")
        return [0.0, 0.0, 0.0, 0.0]



