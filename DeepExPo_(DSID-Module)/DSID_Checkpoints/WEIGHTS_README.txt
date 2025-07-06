=========================
Model Weights Download
=========================

The model weight files for this project are too large to store directly in this GitHub repository.  
They have therefore been uploaded to Google Drive.

----------------------------------------
Download link  
----------------------------------------
https://drive.google.com/drive/my-drive

----------------------------------------
Installation Instructions
----------------------------------------
1. Open the link above (or copy‑paste it into your browser) and download the weight archive,  
   for example **DeepExPo_Weights.zip**.

2. After downloading, extract or move the contained files to the following path,  
   **DeepExPo\DeepExPo_Weights**:

   (Create the folder if it does not already exist.)

3. Verify that the required files are present, for example:
   • model.safetensors  (or *.bin / *.pth)  
   • tokenizer.json  
   • config.json

4. Continue with the normal setup and execution steps described in **README.md**.  
   The code will automatically load the weights from `DeepExPo/DeepExPo_Weights/`.

----------------------------------------
Troubleshooting
----------------------------------------
• **Google Drive quota exceeded** – choose **File → Make a copy** to copy the archive to your own Drive,  
  then download from there.

• **Path errors** – make sure the file names and directory structure match what the code expects.  
  You can adjust paths in the project’s configuration files if necessary.

• **Updated weights** – when you retrain or update the model, upload the new archive to Google Drive  
  and update the link in this file.

----------------------------------------
Created on: 06 July 2025
