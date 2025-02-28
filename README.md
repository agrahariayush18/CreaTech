# AI-Driven Construction Project Monitoring Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-0.12.0-blue)  
![Python](https://img.shields.io/badge/Python-3.10-green)  
![License](https://img.shields.io/badge/License-MIT-purple)  

## Table of Contents
- **Project Overview**  
- **Key Features**  
- **Installation**  
- **Usage**  
- **Deployment**  
- **Contributing**  
- **License**  
- **Contact**  

---

## Project Overview  
This project addresses **delays, cost overruns, and inefficiencies** in large-scale construction projects in India using **AI, computer vision, and predictive analytics**. The solution includes a **real-time dashboard** for monitoring progress, analyzing risks, optimizing resources, and improving decision-making.  

**Problem Statement**:  
- 98% of large Indian construction projects face delays and cost overruns due to unpredictable site conditions, workforce shortages, and supply chain disruptions.  
- Traditional methods lack real-time insights, leading to poor resource allocation and safety hazards.  

**Solution**:  
- A **Streamlit-based dashboard** that integrates computer vision (OpenCV) and machine learning (Scikit-learn) for:  
  - Real-time progress estimation.  
  - Risk prediction.  
  - Workforce and material cost optimization.  
  - Supply chain resilience analysis.  

---

## Key Features 🚀  
✅ **Real-Time Monitoring**:  
- Track construction progress using computer vision (e.g., edge detection).  
- Analyze live video feeds or images for progress estimation.  

✅ **Predictive Analytics**:  
- Forecast delays and cost overruns using historical data.  
- Optimize resource allocation with machine learning.  

✅ **Risk Analysis**:  
- Classify projects into **High/Medium/Low risk** based on weather, workforce, and supply chain data.  

✅ **Supply Chain Resilience**:  
- Monitor material shortages and supplier performance.  

✅ **Cost Efficiency**:  
- Compare budget vs. actual costs and reduce wastage.  

---

## Installation 💻  
### Step 1: Clone the Repository  
```bash  
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  
```  

### Step 2: Set Up a Virtual Environment (Recommended)  
```bash  
python3.10 -m venv venv  # Use Python 3.10 or 3.11 (critical for compatibility)  
source venv/bin/activate  # On Windows: venv\Scripts\activate  
```  

### Step 3: Install Dependencies  
```bash  
pip install -r requirements.txt  
```  

**Note**: If deploying on cloud platforms (e.g., Streamlit Cloud), ensure your `requirements.txt` includes:  
```plaintext  
streamlit==1.22.0  
pandas==2.0.3  
numpy==1.23.5  
opencv-python-headless==4.7.0.72  # Use headless version for cloud compatibility  
scikit-learn==1.3.0  
matplotlib==3.7.1  
seaborn==0.12.2  
```  

---

## Usage 📊  
### Run the Dashboard Locally  
```bash  
streamlit run app.py  
```  

### Key Functionalities:  
1. **Upload a CSV File**:  
   - Use your own dataset or the provided synthetic dataset (`synthetic_data.csv`).  

2. **Real-Time Progress Estimation**:  
   - Upload images of construction sites to estimate progress using computer vision.  

3. **Predictive Analytics**:  
   - Input planned duration to predict progress and cost efficiency.  

---

## Deployment 🌐  
### Streamlit Community Cloud  
1. Push your code to GitHub.  
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud).  
3. Create a new app and link it to your repository.  
4. Specify `Python 3.10` in `runtime.txt` to avoid dependency errors.  

### Heroku  
1. Create a `Procfile`:  
   ```plaintext  
   web: sh setup.sh && streamlit run app.py  
   ```  
2. Create `setup.sh`:  
   ```bash  
   mkdir -p ~/.streamlit/  
   echo "[server]\nheadless = true\nenableCORS = false\nport = \$PORT" > ~/.streamlit/config.toml  
   ```  
3. Deploy via Heroku CLI:  
   ```bash  
   git push heroku main  
   ```  

---

## Project Structure 📂  
```  
your-repo-name/  
├── app.py                # Main Streamlit app  
├── requirements.txt      # Dependencies  
├── synthetic_data.csv    # Synthetic dataset for Indian construction projects  
├── setup.sh              # Configuration for Heroku  
└── Procfile              # Deployment instructions for Heroku  
```  

---

## License 📜  
This project is licensed under the **MIT License**.  

---

## Contact 📧  
For collaboration or feedback, reach out to:  
- **Team Leader**: [Ayush Agrahari ]  
- **Email**: [ajay.agrahari9788gmail.com]  
- **LinkedIn**: [www.linkedin.com/in/ayush-agrahari-1845koro]     

---

### References  
- [Identifying Challenges of Construction Industry in India](https://www.researchgate.net/publication/355445852_Identifying_Challenges_of_Construction_Industry_in_India)  
- [The Benefits of AI in Construction](https://constructible.trimble.com/construction-industry/the-benefits-of-ai-in-construction)  
- [Computer Vision in Construction](https://viso.ai/applications/computer-vision-in-construction/)  
