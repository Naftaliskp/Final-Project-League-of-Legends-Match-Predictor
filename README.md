# League of Legends Match Predictor

## Project Overview
Machine learning model to predict match outcomes in League of Legends using logistic regression with PyTorch. Analyzes in-game statistics to predict win/loss with 55.5% accuracy.

## Key Results
- **Best Model Accuracy**: 55.5% (with L2 regularization)
- **Dataset**: 1000 matches, 8 features
- **AUC Score**: 0.5944
- **Optimal Learning Rate**: 0.05

## Model Architecture
```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)
        return out
```

## Feature Importance
| Rank | Feature | Weight | Impact |
|------|---------|--------|--------|
| 1 | wards_placed | +0.0895 | Most positive |
| 2 | gold_earned | +0.0693 | Very positive |
| 3 | damage_dealt | -0.0534 | Negative |
| 4 | kills | +0.0267 | Positive |
| 5 | deaths | +0.0117 | Positive |

## Performance Comparison
| Model | Training Acc | Test Acc | Improvement |
|-------|--------------|----------|-------------|
| Baseline | 51.88% | 52.50% | - |
| With L2 Regularization | 52.75% | 55.50% | +3.00% |

## Technologies
- Python 3.9+
- PyTorch 2.8.0
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- Jupyter Notebook

## Project Structure
```
League-of-Legends-Match-Predictor/
├── league_predictor.ipynb         # Main notebook
├── league_of_legends_model.pth    # Saved model weights
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── data/
│   └── league_of_legends_data_large.csv  # Dataset
└── images/                        # Visualization outputs
```

## Quick Start
```bash
# Clone repository
git clone https://github.com/Naftaliskp/League-of-Legends-Match-Predictor.git

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook league_predictor.ipynb
```

## Complete Pipeline
1. Data loading & preprocessing
2. Logistic regression implementation
3. Model training & evaluation
4. L2 regularization optimization
5. Confusion matrix & ROC analysis
6. Hyperparameter tuning (learning rate)
7. Feature importance analysis
8. Model saving & loading

## Visualizations
![Loss Curves](images/loss_curves.png)
*Training and test loss convergence*

![ROC Curve](images/roc_curve.png)
*Receiver Operating Characteristic curve*

![Feature Importance](images/feature_importance.png)
*Top predictive features for match outcomes*

## Dependencies
```
torch==2.8.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
```

## Key Findings
1. **Wards placed** is the strongest predictor of victory
2. **Gold advantage** significantly increases win probability
3. **High damage dealt** may indicate inefficient playstyle
4. **L2 regularization** improves generalization by 3%
5. **Learning rate 0.05** provides optimal convergence

## Future Enhancements
- Expand dataset with more match data
- Implement neural networks for non-linear patterns
- Add champion-specific features
- Create real-time prediction API
- Develop interactive dashboard

---

*Predicting League of Legends match outcomes using machine learning and PyTorch*
```
