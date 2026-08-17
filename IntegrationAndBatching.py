from collections import defaultdict 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
    
    
processed_dataset = []
for example in dataset["train"]:
    processed = preprocess_example(example, label_to_prototype, tokenizer, image_transform)
    processed_dataset.append(processed) 
    
        
sample = processed_dataset[0]
print("🔍 Sample keys and types:")
for key, value in sample.items():
    print(f"  {key}: {type(value)}", end="")
    if isinstance(value, torch.Tensor):
        print(f" | shape: {value.shape}")
    elif isinstance(value, dict):
        print(f" | keys: {list(value.keys())}")
    else:
        print(f" | value: {str(value)[:50]}")

for i, sample in enumerate(processed_dataset):
    missing = [k for k in sample if sample[k] is None]
    if missing:
        print(f"⚠️ Sample {i} missing: {missing}")
    
unknowns = [ex for ex in processed_dataset if ex['hazard_prototype'] == "unknown"]
print(f"🚨 {len(unknowns)} examples have unknown hazard prototypes.")

question_shapes = set(ex['question_input_ids'].shape for ex in processed_dataset)
image_shapes = set(ex['image'].shape for ex in processed_dataset)

print("🧪 Question input shapes:", question_shapes)
print("🖼️ Image tensor shapes:", image_shapes)


for i, ex in enumerate(processed_dataset[:5]):
    print(f"\n🧠 Sample {i + 1} Semantics:")
    print(f"  Raw: {ex['question_semantics']['raw']}")
    print(f"  Entities: {ex['question_semantics']['entities']}")
    print(f"  Actions: {ex['question_semantics']['actions']}")
    

column_summary = defaultdict(set)
for ex in processed_dataset:
    for key, value in ex.items():
        if isinstance(value, torch.Tensor):
            column_summary[key].add(value.shape)
        elif isinstance(value, dict):
            column_summary[key].add(tuple(value.keys()))
        else:
            column_summary[key].add(type(value))

print("📊 Column Summary:")
for key, summary in column_summary.items():
    print(f"  {key}: {summary}")


def custom_collate_fn(batch):
    input_ids = [item['question_input_ids'] for item in batch]
    attention_mask = [item['question_attention_mask'] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    images = torch.stack([item['image'] for item in batch])
    hazard_prototypes = [item['hazard_prototype'] for item in batch]
    semantics = [item['question_semantics'] for item in batch]

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'images': images,
        'hazard_prototype': hazard_prototypes,
        'question_semantics': semantics
    }
    
#### The data loader with custom_collate_fn
train_loader = DataLoader(
    processed_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn
)

for batch in train_loader:
    print("✅ Batch keys:", batch.keys())
    print("📏 input_ids shape:", batch['input_ids'].shape)
    print("📏 attention_mask shape:", batch['attention_mask'].shape)
    print("🖼️ images shape:", batch['images'].shape)
    print("🧪 hazard_prototype:", batch['hazard_prototype'][:3])  # show first 3
    print("🧠 question_semantics sample:", batch['question_semantics'][0])
    break  # just one batch

