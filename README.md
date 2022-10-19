# deepneumo-service
This service takes a chest X-ray (CXR) scan and returns the set of predictions that is guaranteed to contain the true diagnosis at a specific significance level using conformal predictions.

# About
Pneumonia is a common acute respiratory infection that affects tge alveoli and distal airways. It is often cause of major health problems and is associated with high morbidity and both short- and long-term mortality if all age groups worldwide. [1]

This project implements EfficientNet-B5 with cutmix augmentation as proposed by Nishio et al.[2] using PyTorch.

The model will be traindef using the RSNA Pneumonia Detection Challenge, available in raw form from [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data).

The model will be finally wrapped in a microservice and deployed to a Kubernetes cluster using Seldon. Users will be able to send post requests for online predictions (both single and batch requests) and receive classification for the CXR.

**[Note] This is an on-going project. Check out the `dev` branch for updates.**

# Roadmap
- [ ] Three-way split of dataset (train, val, and test).
- [ ] Custom DicomDataset and DataLoader
- [ ] CircleCI for automated tests
- [ ] Code coverage using Codecov
- [ ] Automated package release workflow using semantic-release
- [ ] MLFlow integration for model tracking
- [ ] Model training
- [ ] Add conformal prediction
- [ ] Model internal validation
- [ ] Model external validation
- [ ] Deploy using Seldon

# References
[1] Torres, A., Cilloniz, C., Niederman, M.S. et al. Pneumonia. Nat Rev Dis Primers 7, 25 (2021). https://doi.org/10.1038/s41572-021-00259-0
[2] Nishio, M., Kobayashi, D., Nishioka, E. et al. Deep learning model for the automatic classification of COVID-19 pneumonia, non-COVID-19 pneumonia, and the healthy: a multi-center retrospective study. Sci Rep 12, 8214 (2022). https://doi.org/10.1038/s41598-022-11990-3
