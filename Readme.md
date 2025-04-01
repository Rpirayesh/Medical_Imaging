## 🔧 Requirements
This code is designed to run in **Google Colab**. It uses:
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

No installation needed if you use Colab — just upload and run!

---

## 📂 How to Use
1. Open the notebook in **Google Colab**
2. Upload multiple ultrasound videos for training 
3. The pipeline will:
   - Preprocess frames
   - Extract features
   - Train an SVM classifier
4. Upload a new video for testing
5. The system will classify whether free fluid is detected or not
