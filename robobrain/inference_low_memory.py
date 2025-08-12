import torch
import gc
import os
from transformers import AutoProcessor, AutoModelForPreTraining, BitsAndBytesConfig
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class LowMemoryInference:
    """
    Ultra memory-efficient inference class using quantization and other optimizations.
    Use this when facing severe CUDA memory constraints.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain", lora_id=None, device="cuda:0", use_8bit=True, use_4bit=False):
        """
        Initialize with aggressive memory optimizations.
        
        Args:
            model_id (str): Model identifier
            lora_id (str, optional): LoRA weights path
            device (str): Target device
            use_8bit (bool): Use 8-bit quantization
            use_4bit (bool): Use 4-bit quantization (most aggressive)
        """
        self.device = device
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        
        # Set environment variables for memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Clear all GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
        
        print("Loading model with quantization...")
        self._load_model_with_quantization(model_id, lora_id)
        
    def _load_model_with_quantization(self, model_id, lora_id):
        """Load model with quantization configuration"""
        
        # Configure quantization
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("Using 4-bit quantization")
        elif self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            print("Using 8-bit quantization")
        
        try:
            # Load model with quantization
            self.model = AutoModelForPreTraining.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=False,
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
            # Load LoRA if specified
            if lora_id is not None:
                from peft import PeftModel
                print("Loading LoRA weights...")
                self.processor.image_processor.image_grid_pinpoints = [[384, 384]]
                if hasattr(self.model, 'base_model'):
                    self.model.base_model.base_model.config.image_grid_pinpoints = [[384, 384]]
                self.model = PeftModel.from_pretrained(self.model, lora_id)
                print(f"Model loaded with LoRA: {lora_id}")
            
            print("Model loaded successfully!")
            self.check_memory()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying fallback loading method...")
            self._fallback_load(model_id, lora_id)
    
    def _fallback_load(self, model_id, lora_id):
        """Fallback loading without quantization"""
        print("Loading without quantization as fallback...")
        
        self.model = AutoModelForPreTraining.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_cache=False,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        if lora_id is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_id)
    
    def clear_memory(self):
        """Aggressive memory clearing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
    def check_memory(self):
        """Check GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            peak = torch.cuda.max_memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"GPU Memory Usage:")
            print(f"  Allocated: {allocated:.2f}GB")
            print(f"  Cached: {cached:.2f}GB") 
            print(f"  Peak: {peak:.2f}GB")
            print(f"  Total: {total:.2f}GB")
            print(f"  Free: {total - allocated:.2f}GB")
            
            return allocated, cached, total
        return 0, 0, 0
    
    def inference(self, text, image, max_new_tokens=100, do_sample=True, temperature=0.7):
        """Memory-efficient inference"""
        
        try:
            # Pre-inference cleanup
            self.clear_memory()
            print("Starting inference...")
            
            # Prepare messages
            if isinstance(image, str) and image.startswith("http"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image", "url": image},
                        ],
                    },
                ]
            else:
                if isinstance(image, str):
                    image = Image.open(image)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image", "image": image},
                        ],
                    },
                ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate with memory constraints
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=False,
                    return_dict_in_generate=False,
                )
            
            # Decode
            prediction = self.processor.decode(
                output[0][2:],
                skip_special_tokens=True
            ).split("assistant")[-1].strip()
            
            # Cleanup
            del inputs, output
            self.clear_memory()
            
            return prediction
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM Error: {e}")
            self.clear_memory()
            
            # Try with even smaller tokens
            if max_new_tokens > 20:
                print(f"Retrying with max_new_tokens={max_new_tokens//2}")
                return self.inference(text, image, max_new_tokens//2, do_sample, temperature)
            else:
                raise Exception("Cannot run inference even with minimal parameters")
        
        except Exception as e:
            print(f"Inference error: {e}")
            self.clear_memory()
            raise

def run_memory_test():
    """Test different memory configurations"""
    print("Testing memory configurations...")
    
    configurations = [
        {"use_4bit": True, "use_8bit": False, "name": "4-bit quantization"},
        {"use_4bit": False, "use_8bit": True, "name": "8-bit quantization"},
        {"use_4bit": False, "use_8bit": False, "name": "No quantization"},
    ]
    
    for config in configurations:
        print(f"\n--- Testing {config['name']} ---")
        try:
            model = LowMemoryInference(
                "BAAI/RoboBrain", 
                use_4bit=config["use_4bit"],
                use_8bit=config["use_8bit"]
            )
            
            prompt = "What is shown in this image?"
            image = "http://images.cocodataset.org/val2017/000000039769.jpg"
            
            result = model.inference(prompt, image, max_new_tokens=50)
            print(f"Success with {config['name']}")
            print(f"Result: {result}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            gc.collect()
            break
            
        except Exception as e:
            print(f"Failed with {config['name']}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

if __name__ == "__main__":
    # Check available GPU memory first
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Available GPU memory: {total_memory:.2f} GB")
        
        if total_memory < 8:
            print("Low GPU memory detected. Using aggressive optimizations.")
            run_memory_test()
        else:
            print("Sufficient GPU memory. Using standard 8-bit quantization.")
            model = LowMemoryInference("BAAI/RoboBrain", use_8bit=True)
            
            prompt = "What is shown in this image?"
            image = "http://images.cocodataset.org/val2017/000000039769.jpg"
            
            pred = model.inference(prompt, image, max_new_tokens=100)
            print(f"Prediction: {pred}")
    else:
        print("CUDA not available. Please check your GPU setup.")
