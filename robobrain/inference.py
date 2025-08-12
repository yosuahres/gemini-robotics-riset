import torch
import gc
from transformers import AutoProcessor, AutoModelForPreTraining
from PIL import Image
import warnings

class SimpleInference:
    """
    A class for performing inference using Hugging Face models with optional LoRA adapters.
    Supports both local images and image URLs as input.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain", lora_id=None, device="cuda:0", dtype=torch.float16):
        """
        Initialize the model and processor with memory optimizations.
        
        Args:
            model_id (str): Path or Hugging Face model identifier (default: "BAAI/RoboBrain")
            lora_id (str, optional): Path or Hugging Face model for LoRA weights. Defaults to None.
            device (str): Device to load model on (default: "cuda:0")
            dtype: Data type for model weights (default: torch.float16)
        """
        self.device = device
        self.dtype = dtype
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print("Loading Checkpoint ...")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load model with memory optimizations
        self.model = AutoModelForPreTraining.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map="auto",  # Automatically distribute across available GPUs
            trust_remote_code=True,
            # Enable memory optimizations
            use_cache=False,  # Disable KV cache to save memory
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # If LoRA weights are provided, load and adapt the base model
        if lora_id is not None:
            from peft import PeftModel
            print("Loading LoRA Weights...")
            self.processor.image_processor.image_grid_pinpoints = [[384, 384]]
            self.model.base_model.base_model.config.image_grid_pinpoints = [[384, 384]]
            self.model = PeftModel.from_pretrained(self.model, lora_id)
            print(f"Model is initialized with {model_id} and {lora_id}.")
        else:
            print(f"Model is initialized with {model_id}.")
            
        # Enable gradient checkpointing to save memory during inference
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            
        print(f"GPU memory after loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    def check_memory(self):
        """Check current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB")
            return allocated, cached, total
        return 0, 0, 0
        
    def inference(self, text, image, do_sample=True, temperature=0.7, max_new_tokens=250):
        """
        Perform inference with text and image input, with memory management.
        
        Args:
            text (str): Input text prompt
            image: Image input (PIL Image, file path, or URL)
            do_sample (bool): Whether to use sampling
            temperature (float): Sampling temperature
            max_new_tokens (int): Maximum number of tokens to generate
        """
        try:
            # Clear memory before inference
            self.clear_memory()
            
            # Check memory before processing
            print("Memory usage before inference:")
            self.check_memory()
            
            # Process image input
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
            elif isinstance(image, Image.Image) or isinstance(image, str):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image", "image": Image.open(image) if isinstance(image, str) else image},
                        ],
                    },
                ]

            print("Processing input...")
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Move inputs to device with error handling
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            print("Generating output...")
            
            # Use context manager to ensure no gradients are computed
            with torch.no_grad():
                # Enable attention slicing to reduce memory usage
                if hasattr(self.model, 'enable_attention_slicing'):
                    self.model.enable_attention_slicing()
                
                # Generate with memory-efficient settings
                output = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=do_sample, 
                    temperature=temperature,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    # Memory optimization parameters
                    use_cache=False,  # Disable KV cache
                    return_dict_in_generate=False,
                )
            
            # Decode output
            prediction = self.processor.decode(
                output[0][2:],
                skip_special_tokens=True
            ).split("assistant")[-1].strip()
            
            # Clean up intermediate tensors
            del inputs, output
            self.clear_memory()
            
            print("Memory usage after inference:")
            self.check_memory()

            return prediction
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA Out of Memory Error: {e}")
            print("Clearing GPU cache and retrying with reduced parameters...")
            
            # Clear all GPU memory
            self.clear_memory()
            
            # Try again with reduced token count
            if max_new_tokens > 50:
                print(f"Retrying with max_new_tokens={max_new_tokens//2}")
                return self.inference(text, image, do_sample, temperature, max_new_tokens//2)
            else:
                raise Exception("Unable to run inference even with reduced parameters. Consider using a smaller model or more GPU memory.")
                
        except Exception as e:
            print(f"Error during inference: {e}")
            self.clear_memory()
            raise


if __name__ == "__main__":
    
    # Initialize model with memory monitoring
    try:
        print("Initializing model...")
        model = SimpleInference("BAAI/RoboBrain")
        
        prompt = "What is shown in this image?"
        image = "http://images.cocodataset.org/val2017/000000039769.jpg"

        print("\nRunning inference...")
        pred = model.inference(prompt, image, do_sample=True, max_new_tokens=150)
        print(f"Prediction: {pred}")
        
        # Clean up after inference
        model.clear_memory()
        print("\nFinal memory state:")
        model.check_memory()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Reduce max_new_tokens (try 50-100)")
        print("2. Use torch.float32 instead of float16 if you have enough memory")
        print("3. Try using CPU inference by setting device='cpu'")
        print("4. Consider using a smaller model variant")
        print("5. Close other GPU-intensive applications")
        
        # Show current GPU memory state
        if torch.cuda.is_available():
            print(f"\nGPU Memory Info:")
            print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")