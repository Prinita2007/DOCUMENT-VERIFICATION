# Intelligent Academic Document Verification and Fraud Detection Using Machine Learning and Explainable AI

**Research paper draft — dataset-dependent experimental claims are intentionally pending.**

## Abstract

Academic credential review requires careful human judgment: low-quality scans, repeated uploads, OCR errors, inconsistent fields, and altered text can all create review burden. This work presents a secure document-review prototype and a reproducible research pipeline combining document preprocessing, OCR, cross-document rules, duplicate detection, anomaly detection, supervised classifiers, a convolutional neural network, and SHAP-based explanation. The system reports risk indicators and preserves a human administrator's decision; it does not claim that machine learning proves authenticity or fraud. The implementation provides Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and Isolation Forest training with group-aware splits and cross-validation, plus a custom CNN and SHAP permutation explanations. Relevant data sources were reviewed, but access is restricted or unverified and no labeled academic certificate dataset is present. Consequently, no experimental accuracy or trained-model claims are reported in this draft.

**Keywords—** document image analysis, academic credentials, anomaly detection, OCR, explainable AI, human review, dataset shift.

## I. Introduction

Academic institutions process marks cards, transcripts, and transfer certificates that vary by issuing authority, year, scan quality, and language. Automated document checks can prioritize manual inspection, but generic document and identity-card datasets do not represent the institutional rules and template diversity of academic records. A responsible system must expose evidence and uncertainty rather than produce a binary accusation.

This project builds an intake and review application and a reproducible evaluation path. Its central research question is whether engineered image features, supervised classifiers, anomaly detection, and a CNN can provide useful triage signals on properly licensed and labeled document data without conflating adjacent tasks with academic fraud.

## II. Problem Statement and Objectives

The problem is to help authorized college staff organize document submissions and identify records that merit closer review. Objectives are: validate image/PDF uploads; extract text and confidence when Tesseract is configured; compare OCR fields with student records; detect exact duplicates; compute explainable quality/rule/anomaly indicators; preserve administrator review history; train and compare five classical algorithms; provide CNN image analysis and SHAP explanations; and report held-out results only from authorized data.

## III. Related Work

DocTamper studies tampered-text localization in document images and publishes an access-controlled dataset with non-commercial restrictions. MIDV-2020 provides mock identity documents and annotations for document detection/recognition under a license-request process. RVL-CDIP supports document-type classification rather than authenticity or tampering classification. The distinction matters: identity-card fraud, document-type recognition, and academic certificate manipulation are separate targets with different threat models and labels. Isolation Forest provides unsupervised outlier detection; SHAP provides feature-attribution explanations for model predictions; Tesseract provides OCR, not authenticity verification.

## IV. Dataset and Licensing

The verified source review is recorded in `datasets/dataset_sources.csv`:

1. **DocTamper:** official repository reachable; data requires an academic request/access password and is non-commercial. Labels describe manipulated text and masks, not academic-fraud findings. No files were downloaded.
2. **MIDV-2020:** official page describes mock identity documents and requires an access request and acceptance of the dataset license. Its reported 124 GB size and identity-document domain make it adjacent research data, not academic certificate ground truth. No files were downloaded.
3. **RVL-CDIP:** document-type labels do not imply tampering. Its linked old CMU page redirected during inspection and current official access/license terms were not verified. Do not use until terms are confirmed.

All source images retain their applicable rights. No rehosting or redistribution is intended. An academic-use request or license acceptance must be completed by an authorized user before processing the restricted data.

## V. Methodology

### A. Data preparation

The canonical manifest contains a relative image path, binary label (`authentic` / `tampered`), and source `group_id`. Exact SHA-256 duplicates are removed from the processed manifest; conflicting labels stop training. Missing paths and unreadable images are recorded in a rejection report. OpenCV derives file-size logarithm, dimensions, aspect ratio, grayscale moments, Laplacian variance, edge density, and entropy. Median imputation, standardization, and SelectKBest feature selection are fitted only inside each cross-validation training fold through an sklearn pipeline.

Train, validation, and test partitions are disjoint by source group to prevent scans, crops, or manipulated variants of the same original document from crossing partitions. Training fails if there are too few independent groups or a split lacks a class. A fixed seed controls splits and estimators.

### B. Classical algorithms

1. **Logistic Regression:** \(p(y=1|x)=1/(1+e^{-(w^Tx+b)})\), fit by regularized log-loss minimization.
2. **Decision Tree:** recursive feature-threshold splits minimize class impurity; maximum depth and minimum leaf size are tuned.
3. **Random Forest:** aggregate predictions from bootstrap-trained, feature-randomized trees to reduce variance.
4. **Support Vector Machine:** maximize the separating margin; linear and radial-basis kernels and regularization parameter are tuned.
5. **Isolation Forest:** random partition depth estimates isolation; the model is fit only to authentic training examples. Contamination is selected on validation labels, not the held-out test set.

GridSearchCV and three-fold StratifiedGroupKFold select supervised hyperparameters using training groups. Evaluation reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix, training time, and per-example prediction time on the final held-out group split.

### C. CNN image model

The prototype includes a custom PyTorch three-block CNN with convolution, batch normalization, ReLU, pooling, adaptive pooling, dropout, and a binary logit. Images are resized to a fixed RGB tensor; small rotations are used for training augmentation. Validation F1 selects the checkpoint with early stopping. The model is not pretrained. Training requires labeled and licensed images, so no weights are currently available.

### D. Explainability

The selected supervised estimator is explained using a SHAP permutation explainer with a background sampled from training data and a bounded sample from the held-out test set. Mean absolute feature attributions and a bar chart are saved. These values attribute model output under the selected feature representation; they do not prove causation or tampering.

### E. End-to-end verification

The web workflow validates content and size, stores a randomized filename outside static frontend files, extracts native PDF text or invokes preprocessed Tesseract OCR, compares recognized name/registration fields, checks SHA-256 duplicates, computes quality indicators, and runs an Isolation Forest only after enough manually verified examples exist. A trained CNN can be incorporated when an artifact exists. The administrator reviews the original, records a decision, and the backend persists the event. An automated score alone never sets a final fraud decision.

## VI. Architecture and Implementation

React/Vite communicates with FastAPI REST endpoints. Argon2 hashes administrator passwords; expiring signed bearer tokens and role checks protect API access. SQLAlchemy maps students, documents, results, histories, and audit events. PostgreSQL is the configured primary database and Alembic provides schema migration; SQLite is an isolated testing fallback. Documents are stored outside the static web root and require an authenticated download endpoint.

## VII. Experimental Results

**No dataset-backed results are available at this stage.** Restricted dataset access has not been requested or granted, RVL-CDIP licensing/access was not verified, and no labeled manifest exists. Therefore, the following are not reported: model accuracies, precision/recall/F1, ROC-AUC, confusion matrices, SHAP plots, CNN performance, or claimed comparative winners. Run `ml.train_classical` and `ml.train_cnn` after obtaining a license and constructing a source-grouped manifest; generated test metrics will then be available under `results/evaluation/`. Do not fill this section with values from synthetic unit tests.

## VIII. Limitations and Ethical Considerations

There is no representative academic-certificate dataset, no issuer-record integration, and no demonstrated generalization across institutions or regions. OCR can misread names, dates, and marks. Exact hashes miss visually similar scans. Metadata and image quality are weak clues. Identity-document corpora differ from academic documents. A false positive can unfairly harm a student; decisions require source verification, staff oversight, appeal, privacy protection, least-privilege access, and retention controls. Sensitive student records must not be used in a demo or research dataset without institutional approval and appropriate legal basis.

## IX. Conclusion and Future Work

The deliverable is a working review prototype and a reproducible, leakage-aware training/evaluation framework, not a validated fraud classifier. Next steps are dataset permission, lawful labels, Tesseract installation, PostgreSQL credential configuration, real training and external validation on institution-approved academic records, fairness/false-positive review, and a documented human appeal process.

## References

[1] C. Qu et al., “Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution,” CVPR, 2023. [Official repository](https://github.com/qcf-568/DocTamper).  
[2] K. Bulatov et al., “MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis,” *Computer Optics*, vol. 46, no. 2, 2022, doi: 10.18287/2412-6179-CO-1006.  
[3] A. W. Harley, A. Ufkes, and K. G. Derpanis, “Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval,” ICDAR, 2015, doi: 10.1109/ICDAR.2015.7333918.  
[4] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” IEEE ICDE, 2008, doi: 10.1109/ICDE.2008.17.  
[5] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS, 2017.  
[6] R. Smith, “An Overview of the Tesseract OCR Engine,” ICDAR, 2007, doi: 10.1109/ICDAR.2007.4376991.
