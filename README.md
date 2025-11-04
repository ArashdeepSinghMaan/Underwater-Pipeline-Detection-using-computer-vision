# 🚀 Enhancing Underwater Pipeline Detection for Robotic Exploration and Maintenance  
### Using Classical Computer Vision Techniques  

#### 🧠 Authors  
- **Kranti Prakash** – IIT Jodhpur  
- **Chaitanya Shashikant Patil** – IIT Jodhpur  
- **Arashdeep Singh** – IIT Jodhpur  
- **Harshit Rajesh Danorkar** – IIT Jodhpur  

---

## 📖 Overview  

This project presents a **lightweight and efficient classical computer vision pipeline** for detecting **underwater pipelines, cables, and deformable structures** — without relying on deep learning.  
The system is designed for **real-time robotic exploration and maintenance**, particularly on **Autonomous Underwater Vehicles (AUVs)** operating in challenging visibility and lighting conditions.

Unlike deep-learning-based methods, this approach:  
- Requires **no labeled datasets**,  
- Runs on **low-power embedded systems**, and  
- Achieves **robust performance** in **real-world underwater images**.

---

## 🧩 Key Features  

- 🌊 **Purely Classical Vision Pipeline** (no neural networks)  
- ⚙️ **Adaptive Preprocessing** for color correction and contrast enhancement  
- 🔍 **CLAHE + Adaptive Thresholding + Contour Detection**  
- 📏 **Probabilistic Hough Transform** for line detection  
- 🪶 **Lightweight & Real-time** for AUV integration  
- 💡 **Improved Detection Accuracy (83.33%)** compared to baseline (56.25%)  

---

## 📁 Project Structure  

```plaintext
Underwater-Pipeline-Detection/
│
├── data/
│   ├── sample_images/        # Underwater test images
│   ├── results/              # Output of detection pipeline
│
├── src/
│   ├── preprocessing.py      # White balancing, CLAHE, Gaussian filtering
│   ├── edge_detection.py     # Adaptive thresholding and contour extraction
│   ├── hough_transform.py    # Line detection and visualization
│   ├── pipeline.py           # Full algorithm integration
│
├── notebooks/
│   ├── underwater_detection.ipynb   # Google Colab notebook (add your link below)
│
├── README.md
└── requirements.txt
```
---

## 🧮 Methodology  

### **1️⃣ Preprocessing**
- Convert to **Grayscale**  
- Apply **White Balancing**  
- Enhance local contrast using **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  

### **2️⃣ Edge Enhancement**
- Apply **Gaussian Filtering** to reduce noise  
- Use **Adaptive Thresholding** for dynamic edge detection  

### **3️⃣ Feature Extraction**
- Detect and filter **Contours**  
- Apply **ROI selection** to focus on likely pipeline areas  

### **4️⃣ Line Detection**
- Use **Probabilistic Hough Transform** to extract line-like pipeline structures  

### **5️⃣ Post-processing**
- Overlay detected lines on the original image  
- Output visual results for comparison  

---

## 📊 Results  

| Algorithm | Images Processed | Successful Detections | Success Rate |
|------------|------------------|------------------------|---------------|
| Existing (Baseline) | 16 | 9 | 56.25% |
| **Proposed (Enhanced)** | **18** | **15** | **83.33%** |

- Enhanced visibility in low-light, turbid underwater conditions  
- Robust to noise and variable illumination  
- Real-time performance suitable for AUV deployment  

---

## 🧪 Example Output  

| Original Image | 
![Original Image](https://github.com/ArashdeepSinghMaan/Enhancing-Underwater-Pipeline-Detection-for-Robotic-Exploration-and-Maintenance-Using-Classical-Comp/blob/a3b183af526597c546e3fddfb6e4437e190470a7/data/sample_images/IMG-20240307-WA0031.jpg)

| Enhanced Result |
![Enhanced Detection Result](https://github.com/ArashdeepSinghMaan/Enhancing-Underwater-Pipeline-Detection-for-Robotic-Exploration-and-Maintenance-Using-Classical-Comp/blob/a3b183af526597c546e3fddfb6e4437e190470a7/data/results/output14.png)



---

## 🧰 Requirements  

```bash
opencv-python
numpy
matplotlib
scikit-image
```
🏁 Citation

If you use this work, please cite:

@inproceedings{singh2025pipeline,
  title={Enhancing Underwater Pipeline Detection for Robotic Exploration and Maintenance Using Classical Computer Vision Techniques},
  author={Prakash, Kranti and Patil, Chaitanya Shashikant and Singh, Arashdeep and Danorkar, Harshit Rajesh},
  booktitle={Proceedings of [Conference Name]},
  year={2025},
  organization={IIT Jodhpur}
}
