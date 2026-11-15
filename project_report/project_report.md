# Project Report

## Intelligent Academic Document Verification and Fraud Detection Using Machine Learning and Explainable AI

**Status:** implementation and research protocol completed; dataset-dependent experiments are pending lawful dataset access. This report does not contain fabricated metrics or claims of model training.

## 1. Abstract

This project implements a web-based academic document review prototype and a reproducible machine-learning pipeline for examining image quality, duplicate uploads, OCR-readable fields, cross-document inconsistencies, and image-feature outliers. The FastAPI backend stores review records in SQLAlchemy-backed databases, uses administrator authentication, and exposes protected document workflows to a React dashboard. The research pipeline includes Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, Isolation Forest, a custom PyTorch CNN, and SHAP explanations. Because DocTamper access is restricted, MIDV-2020 requires an application and license acceptance, RVL-CDIP's current access terms were not verified, and no dataset was present locally, the project reports no supervised accuracy or trained CNN results. All automated outputs are review indicators and do not establish fraud.

## 2. Problem and objectives

Colleges receive scanned or photographed academic records that may contain poor-quality text, duplicates, inconsistent student identifiers, or altered regions. Manual review remains essential, but staff can benefit from a traceable intake and triage tool. Objectives are to validate uploads, extract text when OCR is available, compare fields with student records, record reasons for flags, preserve review history, and provide data-driven research tooling that can later be evaluated on appropriately licensed data.

## 3. Scope and ethical boundary

The software is a prototype for decision support. Neither an anomaly score nor an OCR mismatch is evidence that a student committed fraud. Model alerts must be independently checked against issuing-institution records and original documents. Public identity-document and generic document datasets are not equivalent to academic certificates and must not be presented as academic-fraud labels.

## 4. Dataset review

The source inventory is maintained in `datasets/dataset_sources.csv`. DocTamper is directly relevant to tampered text in document images but is non-commercial/restricted and requires a qualifying academic request; its labels are text-tamper masks, not college certificate fraud. MIDV-2020 contains mock identity documents and requires an access request and license acceptance; its 124 GB reported size and identity-document domain make it an adjacent benchmark, not an academic-certificate dataset. RVL-CDIP is a document-type classification corpus without tampering labels; the linked old CMU page redirected during inspection and its current license/download access could not be verified. No raw dataset has been downloaded or copied.

## 5. System architecture

```text
Administrator browser (React/Vite)
        | authenticated REST API
FastAPI routes and role checks
        | SQLAlchemy / Alembic
PostgreSQL 18 (primary; SQLite isolated test fallback)
        |
Document pipeline: type/size checks -> protected storage -> PDF/image decode
        -> OCR (Tesseract when installed) -> field and consistency rules
        -> exact SHA-256 duplicate check -> Isolation Forest baseline
        -> optional trained CNN -> explainable risk indicators -> human review

Research pipeline: authorized image manifest -> cleaning/features
        -> source-group train/validation/test splits -> tuned classical models
        -> CNN -> held-out metrics, plots, and SHAP explanations
```

## 6. Preprocessing and leakage controls

The dataset manifest requires `image_path`, `label`, and `group_id`. Paths are constrained to the supplied data root. Missing paths, invalid labels, unreadable images, and exact duplicates are recorded as rejected rows; conflicting labels for identical image bytes stop training. Numeric image features include file-size logarithm, image dimensions/aspect ratio, grayscale mean/standard deviation, Laplacian variance, Canny edge density, and grayscale entropy. Missing numeric values are median-imputed, standardized for the classical estimators, and selected using `SelectKBest`. Labels are encoded as authentic=0 and tampered=1. All scans/edits from the same source must share one group ID; groups are disjoint across train, validation, and test. EDA outlier flags are reported, not silently deleted.

## 7. Algorithms and mathematical outline

- **Logistic Regression:** estimates \(P(y=1|x)=\sigma(w^T x+b)\); regularized cross-entropy is tuned by grouped cross-validation.
- **Decision Tree:** recursively partitions feature space using impurity reduction; depth and leaf size are tuned to constrain overfitting.
- **Random Forest:** averages many bootstrapped randomized trees; depth and minimum leaf size are tuned.
- **Support Vector Machine:** maximizes the margin between classes, with linear/RBF kernels and regularization tuned on training groups.
- **Isolation Forest:** isolates points with random feature/split trees; only authentic training examples form the reference baseline. Its contamination is selected on validation labels; held-out test data is never used for tuning.
- **CNN:** a custom three-block convolutional network maps resized RGB pages to a tampering-pattern score. Validation F1 drives early stopping. It is not pretrained and must be trained on licensed, representative images before use.
- **SHAP:** permutation-based SHAP values explain changes in positive-class probability relative to training-set background rows. SHAP is explanatory for the fitted model, not causal evidence.

## 8. Model optimization and evaluation protocol

The training script uses a fixed random seed, group-aware train/validation/test holdout, three-fold `StratifiedGroupKFold`, and `GridSearchCV` on training groups. Validation selects the best supervised model for SHAP reporting and the Isolation Forest contamination. The test split is used only for final evaluation. It records accuracy, precision, recall, F1, ROC-AUC, confusion matrices, cross-validation tables, and timing; EDA and evaluation figures are generated under `results/`. If data are too sparse to support both labels in each split or grouped CV, training stops with an explicit error instead of relaxing leakage protection.

## 9. Results

**No dataset-backed experiment has been run.** There is no approved labeled manifest in this workspace, DocTamper and MIDV-2020 require access/terms, RVL-CDIP availability and license remain unverified, and the required raw examples are absent. Therefore accuracy, precision, recall, F1, ROC-AUC, confusion matrices, SHAP values, and CNN results are **not available**. Do not quote values until the documented training commands produce these artifacts.

## 10. Implementation and security

The FastAPI application includes authenticated routes, Argon2 password hashing, signed expiring tokens, reviewer/admin roles, upload type/size validation, unique file names, protected downloads, audit logging, PostgreSQL-default SQLAlchemy configuration, and Alembic migrations. The current local PostgreSQL service accepts TCP connections but application credentials have not authenticated; no password reset or `pg_hba.conf` change was made. Tesseract's Python binding exists, but the executable is not installed/on PATH, so actual OCR through Tesseract remains unavailable. Tests use an isolated temporary SQLite database and do not establish PostgreSQL persistence.

## 11. Limitations

1. No domain-representative academic-certificate tampering data or institutional labels are available.
2. The current OCR executable is missing; OCR behavior has only been tested for honest unavailable-engine handling.
3. PostgreSQL authentication is unresolved, so the real PostgreSQL integration has not been exercised.
4. Image-derived statistical features and a small CNN can overfit and may not transfer across scanners, document templates, languages, or manipulations.
5. Exact hashes detect byte-identical duplicates only; OCR extraction is heuristic; image metadata and blur are weak indicators.
6. Results from identity documents, generic forms, or document-type datasets cannot support claims of academic fraud detection.

## 12. Future work

Obtain written dataset permissions, prepare source-group labels and masks, collect an institution-approved academic-certificate dataset with privacy safeguards, install/configure Tesseract, connect PostgreSQL with a least-privilege role, execute the reproducible pipelines, externally validate by institution/template, and have reviewers audit false positives and disparate impact. A deployment should require records from issuing institutions and human review before any consequential decision.

## References

1. C. Qu et al., “Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution,” *Proceedings of CVPR*, 2023. [Official project and access terms](https://github.com/qcf-568/DocTamper).
2. K. Bulatov et al., “MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis,” *Computer Optics*, vol. 46, no. 2, 2022, doi: [10.18287/2412-6179-CO-1006](https://doi.org/10.18287/2412-6179-CO-1006).
3. A. W. Harley, A. Ufkes, and K. G. Derpanis, “Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval,” *International Conference on Document Analysis and Recognition*, 2015, doi: [10.1109/ICDAR.2015.7333918](https://doi.org/10.1109/ICDAR.2015.7333918).
4. F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” *IEEE International Conference on Data Engineering*, 2008, doi: [10.1109/ICDE.2008.17](https://doi.org/10.1109/ICDE.2008.17).
5. S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” *Advances in Neural Information Processing Systems*, 2017. [Proceedings](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
6. R. Smith, “An Overview of the Tesseract OCR Engine,” *Proceedings of ICDAR*, 2007, doi: [10.1109/ICDAR.2007.4376991](https://doi.org/10.1109/ICDAR.2007.4376991).
7. Scikit-learn developers, [Pipeline and model-selection documentation](https://scikit-learn.org/stable/modules/grid_search.html).
