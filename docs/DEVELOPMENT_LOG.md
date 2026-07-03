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
- ***

## Sprint 2

### Objective

Transform the machine learning implementation into a professional, version-controlled software project.

### Completed

- Created modular project structure
- Added configuration management
- Added Git version control
- Published public GitHub repository
- Created project documentation
- Added requirements.txt
- Added CHANGELOG
- Added DEVELOPMENT_LOG
- Prepared project for production deployment

### Outcome

Version 1.0.0 has been successfully published to GitHub.

The project is now maintained as a public software repository and is ready for continued feature development.

### Next Sprint

- FastAPI backend
- REST API
- Interactive API documentation

## Sprint 3

### Objective

Expose the trained XLM-RoBERTa model as a REST API.

### Completed

- Built FastAPI backend
- Added request/response schemas
- Implemented prediction endpoint
- Implemented health endpoint
- Added API information endpoint
- Added root endpoint
- Configured Swagger documentation
- Created automated API tests
- Validated all endpoints successfully

### Outcome

Version 1.1.0 introduces a production-ready REST API capable of serving fake news predictions over HTTP.
