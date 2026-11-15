# Processed data

`ml.train_classical` and `ml.train_cnn` write engineered feature tables, rejected manifest rows, and group-aware split assignments here when run with an approved labeled dataset. These generated files are intentionally not committed. Processing preserves source files and only removes exact duplicate image bytes from the derived manifest.
