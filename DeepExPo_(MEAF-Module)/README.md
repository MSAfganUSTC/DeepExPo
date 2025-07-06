# DeepExPo

**Facial Expression and Pose Generation via Self-Supervised Disentangled Embeddings Fusion in Text-to-Image Diffusion Models**

## 🔍 Overview

**DeepExPo** addresses the limitations of existing text-to-image diffusion models in human-centric generation tasks, specifically facial expression and head pose control. While traditional models can render subjects in various textual contexts, they lack precision in manipulating key human attributes like **facial expression** and **pose** without degrading identity.

To overcome these limitations, DeepExPo introduces a two-stage solution:
- **Dynamic Semantic and Identity Disentanglement (DSID)** module  
- **Multi-Embedding Attentiob Fusion (MEAF)** module

This repository includes the full implementation of the **MEAF module**.

### Requirements
* All testing and development was conducted on 4 NVIDIA GeForce RTX 4090 GPUs, equipped with 24GB of VRAM. 
* 64-bit Python 3.8 and PyTorch 2.1 (or later). See  [https://pytorch.org](https://pytorch.org/)  for PyTorch install instructions.

For convenience, a `requirements.txt` file is included to install the required dependencies in an environment of your choice.

### Usage

A sample training script is provided which assumes a pre-trained diffusion UNet is available for use. If not, one can be trained using the utilities provided in the repository, or following the simpleDiffusion paradigm implemented in faverogian/simpleDiffusion. 

	from diffusion.unet import UNet_Embedding, Multi_Embedding
	from diffusion.Multi_Embedding_net import Multi_Embedding
	from utils.canny import AddCannyImage
	
	from datasets import load_dataset
	from torchvision import transforms
	import torch
	from diffusers.optimization import get_cosine_schedule_with_warmup
	import numpy as np

	class TrainingConfig:
	    image_size = 512  # the generated image resolution
	    train_batch_size = 4
	    num_epochs = 1000
	    gradient_accumulation_steps = 1
	    learning_rate = 5e-5
	    lr_warmup_steps = 10000
	    save_image_epochs = 50
	    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
	    output_dir="DeepExPo/DeepExPo_Weights/DeepExPo_MEAF_weights"  # the model save locally
	    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
	    seed = 0


	def main():   
	
	    config = TrainingConfig
	
	    dataset_name = "MSAFGAN"
	
	    dataset = load_dataset(dataset_name, split="train")
	
	    preprocess = transforms.Compose(
	        [
	            transforms.Resize((config.image_size, config.image_size)),
	            transforms.RandomHorizontalFlip(),
	            transforms.ToTensor(),
	            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	        ]
	    )
	
	    def transform(examples):
	        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
	        conditions = [AddCannyImage()(image) for image in images]
	        return {"images":images, "conditions":conditions}
	
	    dataset.set_transform(transform)
	
	    train_loader = torch.utils.data.DataLoader(
	        dataset,
	        batch_size=config.train_batch_size,
	        shuffle=True,
	    )
	
	    unet = UNetCondition2D.from_pretrained("DeepExPo/DeepExPo_MEAF_weights/", variant="fp16")
	   Multi_Embedding = Multi_Embedding.from_unet(unet, conditioning_channels=1)
	
	    optimizer = torch.optim.Adam(Multi_Embedding.parameters(), lr=config.learning_rate)
	    lr_scheduler = get_cosine_schedule_with_warmup(
	        optimizer,
	        num_warmup_steps=config.lr_warmup_steps,
	        num_training_steps=len(train_loader) * config.num_epochs,
	    )
	
	    Multi_Embedding_model = Multi_Embedding(
	        unet=unet,
	        multi_Embedding=multi_Embedding,
	        image_size=config.image_size
	    )
	
	    Multi_Embedding_model.train_loop(
	        config=config,
	        optimizer=optimizer,
	        train_dataloader=train_loader,
	        lr_scheduler=lr_scheduler
	    )
	    
	    if __name__ == '__main__':
	        main()

### Multi-GPU Training
The code is equipped with HuggingFace's [Accelerator](https://huggingface.co/docs/accelerate/en/index) wrapper for distributed training. Multi-GPU training is easily done via:
`accelerate launch --multi-gpu train.py`

### Sample Results
The model Weighs of DeepExPo's MEAF module are available at [https://github.com/MSAfganUSTC/DeepExPo/DeepExPo_MEAF_weights/

    
    