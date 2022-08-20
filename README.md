# ** Semantic Context Aware Transformer for Street Fashion Image Segmentation**  
**Author: Dingjie Peng $^*$**     
**$*$ Waseda University**
* * *

We use transformer based method to train a deep model for street fashion image segmentation. 
It includes three attention mechanisms: `Semantic Context Self-Attention`, `Cross Stage Attention` and `Pixel Semantic Attention`
* * *
**Requirements**  
The installation of pytorch should use CUDA.
- pytorch >= 1.8.0  
- numpy  
- scipy  
* * *
## 1. Download free-available dataset for testing
- [ModaNet]() 
- Unzip and put dataset in the folder `data`
* * *
## 2. Download the model weights
- Download from Google drive [model_1200.pth]()
- Put weights in the folder `model_zoo`
* * *
## 3. Evaluation
`python main_test.py`  
* * *
### Declaration
The implementation of the model is all done by Dingjie Peng from Waseda University.  
email: [kefipher9013@asagi.waseda.jp](kefipher9013@asagi.waseda.jp)