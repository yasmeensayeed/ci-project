# Hybrid Clustering using Genetic Algorithm and Particle Swarm Optimization

This project implements a hybrid clustering algorithm that combines the strengths of Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) to improve clustering performance. It also includes standard K-Means, standalone GA, and standalone PSO for comparison. A user-friendly Streamlit-based GUI is included for interactive testing.

## 📌 Features

- Clustering with:
  - K-Means
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
  - Hybrid GA + PSO
- GUI for algorithm selection and visualization
- Fitness evaluation using Silhouette Score
- Support for CSV dataset uploads
- Real-time plotting of clustering results

## 🚀 Technologies Used

- Python 3.9+
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## 📂 Project Structure

📁 CI/ ├── data/ # Sample datasets ├── gui.py # Streamlit GUI ├── hybrid.py # Hybrid GA + PSO logic ├── ga.py # Genetic Algorithm implementation ├── pso.py # Particle Swarm Optimization ├── kmeans.py # K-Means wrapper ├── experiments.py # Testing script with logs and output └── utils.py # Helper functions


---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
```git clone https://github.com/your-username/hybrid-clustering-ga-pso.git
cd hybrid-clustering-ga-pso```

2. Install Requirements
``` pip install -r requirements.txt```


3. Run GUI
```streamlit run gui.py```

4. Run Experiments (CLI)
```python experiments.py```

📊 Sample Results
The best fitness score achieved with the Hybrid model:
Silhouette Score = 0.8467

🧠 Future Improvements
Add more metaheuristics (e.g., Ant Colony, Firefly)

Optimize runtime performance

Enhance GUI usability and design

Add clustering reports and plots export options

🧑‍💻 Team Leader
Ziad-Hany
Team Members
[ Ziad Hasan - Ziad AbdelAlem - Mazen Mohammed - Yasmin Elsaid - Tasnim Taha ]

📜 License
This project is open-source and free to use under the MIT License.



