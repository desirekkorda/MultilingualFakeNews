# Development Log

---

## Sprint 1

### Objective

Reproduce the TALLIP fake news detection model and build a reusable software foundation.

### Completed

- Dataset preparation
- Data cleaning
- Exploratory Data Analysis
- XLM-RoBERTa implementation
- Training pipeline
- Evaluation
- Model checkpointing
- Inference engine
- Batch prediction
- Project modularization

### Key Results

Validation Accuracy:

83.90%

Official Test Accuracy:

80.87%

### Lessons Learned

- Mean pooling outperformed CLS pooling.
- Modular code significantly simplified experimentation.
- Lazy loading improves inference efficiency.

### Next Sprint

- GitHub repository
- Professional documentation
- FastAPI backend