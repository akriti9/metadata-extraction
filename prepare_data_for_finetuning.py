#!/usr/bin/env python3
"""
Clean data preparation script for arXiv dataset fine-tuning.
Converts raw data to instruction format for Llama-3.1-8B-Instruct.
"""

import json
import random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

class DataProcessor:
    def __init__(self, max_tokens=3000):
        self.max_tokens = max_tokens
        self.tokenizer = None
        self.load_tokenizer()
        
    def load_tokenizer(self):
        """Load Llama tokenizer for accurate token counting."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load tokenizer ({e})")
            print("  Using approximation method for token counting")
            self.tokenizer = None
    
    def truncate_text(self, text):
        """Truncate text to max_tokens."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= self.max_tokens:
                return text
            truncated_tokens = tokens[:self.max_tokens]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        else:
            # Approximation: 1 token ≈ 4 characters
            max_chars = self.max_tokens * 4
            return text[:max_chars] if len(text) > max_chars else text
    
    def clean_authors(self, authors):
        """Clean and format authors list."""
        if not authors:
            return None
        
        if isinstance(authors, str):
            return [authors.strip()]
        
        if isinstance(authors, list):
            cleaned = [author.strip() for author in authors if author and author.strip()]
            return cleaned if cleaned else None
        
        return None
    
    def clean_text_field(self, text):
        """Clean text fields (title, abstract, etc.)."""
        if not text or not isinstance(text, str):
            return None
        cleaned = ' '.join(text.strip().split())
        return cleaned if cleaned else None
    
    def create_instruction_sample(self, input_text, ground_truth):
        """Convert to instruction format for fine-tuning."""
        
        # System message - Improved prompt
        system_msg = """You are an expert at extracting structured metadata from research papers. Your task is to carefully read the paper text and extract exactly these 5 fields:

- "id": The arXiv identifier (format: XXXX.XXXXX)
- "authors": Complete list of all authors as an array of strings
- "title": The complete paper title (may span multiple lines)
- "abstract": The full abstract text (usually in a dedicated Abstract section)
- "doi": The DOI if present (format: 10.XXXX/...)

Important guidelines:
- Return a valid JSON object with these exact field names
- Use null for any field that cannot be found
- For authors, include all names in the order they appear
- Extract the complete title even if it spans multiple lines
- Include the entire abstract, not just the first sentence

Return only the JSON object, no additional text."""
        
        # User message
        user_msg = f"Extract information from this research paper:\n\n{input_text}"
        
        # Assistant response (ground truth JSON)
        assistant_msg = json.dumps(ground_truth, ensure_ascii=False, indent=2)
        
        # Full conversation in Llama format
        conversation = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{assistant_msg}<|eot_id|>"""
        
        return conversation
    
    def process_sample(self, sample):
        """Process a single data sample."""
        # Get article text
        article_text = sample.get('article_text', '')
        if not article_text:
            return None
        
        # Truncate to max tokens
        truncated_text = self.truncate_text(article_text)
        
        # Extract and clean ground truth fields
        ground_truth = {
            'id': self.clean_text_field(sample.get('id')),
            'authors': self.clean_authors(sample.get('authors')),
            'title': self.clean_text_field(sample.get('title')),
            'abstract': self.clean_text_field(sample.get('abstract')),
            'doi': self.clean_text_field(sample.get('doi'))
        }
        
        # Create instruction format
        conversation = self.create_instruction_sample(truncated_text, ground_truth)
        
        return {
            'text': conversation,
            'input_text': truncated_text,
            'ground_truth': ground_truth
        }
    
    def split_data(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train/validation/test sets."""
        random.shuffle(data)
        
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        return {
            'train': data[:train_size],
            'validation': data[train_size:train_size + val_size],
            'test': data[train_size + val_size:]
        }
    
    def save_data(self, splits, output_dir):
        """Save processed data in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for split_name, split_data in splits.items():
            
            # Save conversations only (for training)
            conversations = [sample['text'] for sample in split_data]
            with open(output_path / f"{split_name}_conversations.json", 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            
            # Save JSONL format (popular for training)
            with open(output_path / f"{split_name}.jsonl", 'w', encoding='utf-8') as f:
                for sample in split_data:
                    json.dump({'text': sample['text']}, f, ensure_ascii=False)
                    f.write('\n')
            
            # Save full data (for analysis)
            with open(output_path / f"{split_name}_full.json", 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Data saved to: {output_path}")
        return output_path
    
    def process_dataset(self, input_file, output_dir="./processed_data"):
        """Main processing function."""
        print("Starting data processing...")
        print(f"Input: {input_file}")
        print(f"Output: {output_dir}")
        print(f"Max tokens: {self.max_tokens}")
        
        # Load data
        print("\nLoading data...")
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"✓ Loaded {len(raw_data)} samples")
        
        # Process samples
        print("\nProcessing samples...")
        processed_data = []
        failed_count = 0
        
        for sample in tqdm(raw_data, desc="Processing"):
            processed = self.process_sample(sample)
            if processed:
                processed_data.append(processed)
            else:
                failed_count += 1
        
        print(f"✓ Successfully processed: {len(processed_data)}")
        print(f"✗ Failed to process: {failed_count}")
        
        if not processed_data:
            print("❌ No valid samples found!")
            return
        
        # Split data
        print("\nSplitting data...")
        splits = self.split_data(processed_data)
        
        for split_name, split_data in splits.items():
            print(f"  {split_name}: {len(split_data)} samples")
        
        # Save data
        print("\nSaving data...")
        output_path = self.save_data(splits, output_dir)
        
        # Print summary
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total valid samples: {len(processed_data)}")
        print(f"Train: {len(splits['train'])}")
        print(f"Validation: {len(splits['validation'])}")
        print(f"Test: {len(splits['test'])}")
        print(f"\nFiles created in: {output_path}")
        print("- *_conversations.json (for training)")
        print("- *.jsonl (JSONL format)")
        print("- *_full.json (complete data)")


def main():
    # Configuration
    INPUT_FILE = "/home/interns/akriti/random_10000_with_t.json"
    OUTPUT_DIR = "./processed_data"
    MAX_TOKENS = 3000
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process data
    processor = DataProcessor(max_tokens=MAX_TOKENS)
    processor.process_dataset(INPUT_FILE, OUTPUT_DIR)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Clean data preparation script for arXiv dataset fine-tuning.
# Converts raw data to instruction format for Llama-3.1-8B-Instruct.
# """

# import json
# import random
# from pathlib import Path
# from tqdm import tqdm
# from transformers import AutoTokenizer

# class DataProcessor:
#     def __init__(self, max_tokens=3000):
#         self.max_tokens = max_tokens
#         self.tokenizer = None
#         self.load_tokenizer()
        
#     def load_tokenizer(self):
#         """Load Llama tokenizer for accurate token counting."""
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 "meta-llama/Llama-3.1-8B-Instruct",
#                 trust_remote_code=True
#             )
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
#             print("✓ Tokenizer loaded successfully")
#         except Exception as e:
#             print(f"⚠ Warning: Could not load tokenizer ({e})")
#             print("  Using approximation method for token counting")
#             self.tokenizer = None
    
#     def truncate_text(self, text):
#         """Truncate text to max_tokens."""
#         if self.tokenizer:
#             tokens = self.tokenizer.encode(text, add_special_tokens=False)
#             if len(tokens) <= self.max_tokens:
#                 return text
#             truncated_tokens = tokens[:self.max_tokens]
#             return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
#         else:
#             # Approximation: 1 token ≈ 4 characters
#             max_chars = self.max_tokens * 4
#             return text[:max_chars] if len(text) > max_chars else text
    
#     def clean_authors(self, authors):
#         """Clean and format authors list."""
#         if not authors:
#             return None
        
#         if isinstance(authors, str):
#             return [authors.strip()]
        
#         if isinstance(authors, list):
#             cleaned = [author.strip() for author in authors if author and author.strip()]
#             return cleaned if cleaned else None
        
#         return None
    
#     def clean_text_field(self, text):
#         """Clean text fields (title, abstract, etc.)."""
#         if not text or not isinstance(text, str):
#             return None
#         cleaned = ' '.join(text.strip().split())
#         return cleaned if cleaned else None
    
#     def create_instruction_sample(self, input_text, ground_truth):
#         """Convert to instruction format for fine-tuning."""
        
#         # System message
#         system_msg = "You are an expert at extracting structured information from research papers. Extract the id, authors, title, abstract, and doi from the given text. Return the result as a JSON object with these exact fields: \"id\", \"authors\", \"title\", \"abstract\", \"doi\". If a field is not found, use null."
        
#         # User message
#         user_msg = f"Extract information from this research paper:\n\n{input_text}"
        
#         # Assistant response (ground truth JSON)
#         assistant_msg = json.dumps(ground_truth, ensure_ascii=False, indent=2)
        
#         # Full conversation in Llama format
#         conversation = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# {system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>
# {user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# {assistant_msg}<|eot_id|>"""
        
#         return conversation
    
#     def process_sample(self, sample):
#         """Process a single data sample."""
#         # Get article text
#         article_text = sample.get('article_text', '')
#         if not article_text:
#             return None
        
#         # Truncate to max tokens
#         truncated_text = self.truncate_text(article_text)
        
#         # Extract and clean ground truth fields
#         ground_truth = {
#             'id': self.clean_text_field(sample.get('id')),
#             'authors': self.clean_authors(sample.get('authors')),
#             'title': self.clean_text_field(sample.get('title')),
#             'abstract': self.clean_text_field(sample.get('abstract')),
#             'doi': self.clean_text_field(sample.get('doi'))
#         }
        
#         # Create instruction format
#         conversation = self.create_instruction_sample(truncated_text, ground_truth)
        
#         return {
#             'text': conversation,
#             'input_text': truncated_text,
#             'ground_truth': ground_truth
#         }
    
#     def split_data(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#         """Split data into train/validation/test sets."""
#         random.shuffle(data)
        
#         total = len(data)
#         train_size = int(total * train_ratio)
#         val_size = int(total * val_ratio)
        
#         return {
#             'train': data[:train_size],
#             'validation': data[train_size:train_size + val_size],
#             'test': data[train_size + val_size:]
#         }
    
#     def save_data(self, splits, output_dir):
#         """Save processed data in multiple formats."""
#         output_path = Path(output_dir)
#         output_path.mkdir(exist_ok=True, parents=True)
        
#         for split_name, split_data in splits.items():
            
#             # Save conversations only (for training)
#             conversations = [sample['text'] for sample in split_data]
#             with open(output_path / f"{split_name}_conversations.json", 'w', encoding='utf-8') as f:
#                 json.dump(conversations, f, ensure_ascii=False, indent=2)
            
#             # Save JSONL format (popular for training)
#             with open(output_path / f"{split_name}.jsonl", 'w', encoding='utf-8') as f:
#                 for sample in split_data:
#                     json.dump({'text': sample['text']}, f, ensure_ascii=False)
#                     f.write('\n')
            
#             # Save full data (for analysis)
#             with open(output_path / f"{split_name}_full.json", 'w', encoding='utf-8') as f:
#                 json.dump(split_data, f, ensure_ascii=False, indent=2)
        
#         print(f"✓ Data saved to: {output_path}")
#         return output_path
    
#     def process_dataset(self, input_file, output_dir="./processed_data"):
#         """Main processing function."""
#         print("Starting data processing...")
#         print(f"Input: {input_file}")
#         print(f"Output: {output_dir}")
#         print(f"Max tokens: {self.max_tokens}")
        
#         # Load data
#         print("\nLoading data...")
#         with open(input_file, 'r', encoding='utf-8') as f:
#             raw_data = json.load(f)
#         print(f"✓ Loaded {len(raw_data)} samples")
        
#         # Process samples
#         print("\nProcessing samples...")
#         processed_data = []
#         failed_count = 0
        
#         for sample in tqdm(raw_data, desc="Processing"):
#             processed = self.process_sample(sample)
#             if processed:
#                 processed_data.append(processed)
#             else:
#                 failed_count += 1
        
#         print(f"✓ Successfully processed: {len(processed_data)}")
#         print(f"✗ Failed to process: {failed_count}")
        
#         if not processed_data:
#             print("❌ No valid samples found!")
#             return
        
#         # Split data
#         print("\nSplitting data...")
#         splits = self.split_data(processed_data)
        
#         for split_name, split_data in splits.items():
#             print(f"  {split_name}: {len(split_data)} samples")
        
#         # Save data
#         print("\nSaving data...")
#         output_path = self.save_data(splits, output_dir)
        
#         # Print summary
#         print(f"\n{'='*50}")
#         print("PROCESSING COMPLETE")
#         print(f"{'='*50}")
#         print(f"Total valid samples: {len(processed_data)}")
#         print(f"Train: {len(splits['train'])}")
#         print(f"Validation: {len(splits['validation'])}")
#         print(f"Test: {len(splits['test'])}")
#         print(f"\nFiles created in: {output_path}")
#         print("- *_conversations.json (for training)")
#         print("- *.jsonl (JSONL format)")
#         print("- *_full.json (complete data)")


# def main():
#     # Configuration
#     INPUT_FILE = "/home/interns/akriti/random_10000_with_t.json"
#     OUTPUT_DIR = "./processed_data"
#     MAX_TOKENS = 3000
    
#     # Set random seed for reproducibility
#     random.seed(42)
    
#     # Process data
#     processor = DataProcessor(max_tokens=MAX_TOKENS)
#     processor.process_dataset(INPUT_FILE, OUTPUT_DIR)


# if __name__ == "__main__":
#     main()
