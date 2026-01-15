# AI-Based Network Intrusion Detection System (NIDS)

## 📌 Project Overview
This project implements an **AI-Based Network Intrusion Detection System (NIDS)** that detects malicious network traffic in real time using **Machine Learning**.

The system is trained on real network traffic data from the **CIC-IDS dataset** and classifies traffic as:

- ✅ Benign (Normal traffic)
- 🚨 Attack (Malicious traffic)

A **Streamlit dashboard** is used to train the model, visualize performance, and simulate live network traffic.

---

## 🎯 Objectives
- Detect cyber attacks using machine learning  
- Classify network traffic as benign or malicious  
- Visualize model accuracy and performance  
- Simulate live traffic in real time  
- Provide an interactive dashboard for users  

---

## 📂 Dataset Used
- **Dataset:** CIC-IDS (Canadian Institute for Cybersecurity)
- Contains realistic network traffic
- Includes both normal and attack patterns
- Sample size used: ~60,000 records (for faster training)

---

## 🧠 Machine Learning Model
- **Algorithm:** Random Forest Classifier

### Why Random Forest?
- Handles large datasets efficiently  
- Works well with non-linear patterns  
- Robust to noise  
- High accuracy for intrusion detection  

---

## 📊 Features Used
- Destination Port  
- Flow Duration  
- Total Forward Packets  
- Packet Length Mean  
- Active Mean  
- Packets per Second (calculated feature)  

The packet-rate feature helps detect flooding attacks like **DDoS** more accurately.

---


