# Univariate Linear Regression from First Principles

This is an implementation of a **univariate linear regression model** from scratch using Python. It serves as an exploration of how a simple linear regression model can be trained via the utilization of **gradient descent** and elucidates how learning algorithms show signs of intelligence based on learning data.


## Objective

The goal would be to add clarity to the black box nature of large language models by exploring and conceptually understanding their simpler counterparts. This project aims to serve as a rite of initiation into machine learning by highlighting the principles that underpins modern AI.


## Project Structure

```
├── salary_data.csv         # Dataset from Kaggle
├── outputs/
│   ├── salary_vs_experience.png
│   └── terminal_output.png     # Logs from training run
├── utils.py        # Tools necessary for Univariate Linear Regression
├── salary_v_experience_model.py        # Application of Univariate Linear Regression
└── README.md                  # You are here
```

## Dataset

The publicly available Salary vs Experience dataset available at the following URL (https://www.kaggle.com/datasets/krishnaraj30/salary-prediction-data-simple-linear-regression)) was used to train the model. Within it lies continuous numeric features that are ideal for univariate linear regressions.


## Training Configuration

- **Optimizer**: Gradient Descent  
- **Cost Function**: Mean Squared Error  
- **Learning Rate (α)**: `0.001`  
- **Iterations**: `100,000`  
- **Normalization**: Z-score standardization  # Although it wasn’t used in the actual implementation as there was only a single feature which was clean and well scaled


## Results

- The cost function converged steadily, indicating successful implementation of gradient descent.
- Final weights and bias represent a line of best fit through the salary data in the dataset.
- The model generalizes well on this simple dataset hence offering insight into the underlying principles of more complex models.


## Visuals

Graphical outputs include:
- Salary vs Experience plot  
- Terminal logs from training run


## Interpretability

By implementing this algorithm from scratch, we gain:
- Intuition on how learning happens
- Appreciation for convex optimization
- A concrete sense of how weights evolve to minimize cost


## References

- Andrew Ng, [Coursera Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- Kaggle Datasets: [Salary Data](https://www.kaggle.com/datasets)
- Supplementary visuals and explanations available in the full report 


## Repository

The complete implementation—along with code, plots, and outputs—is available here as an open-source companion to the full report.

---

> *"To build something from first principles is to witness the elegance hidden beneath the complexity."*
