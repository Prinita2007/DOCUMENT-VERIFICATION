# Raw dataset storage

Place only source datasets that you are authorized to use here. Do not commit raw datasets or student records. The project does not download, mirror, or redistribute the datasets listed in `../dataset_sources.csv`.

The training scripts consume a separate CSV manifest with `image_path,label,group_id` columns. Keep one stable `group_id` for every original document and all of its derived scans, edits, frames, or crops so related samples cannot cross train/validation/test splits. Convert source annotations into the manifest only after checking the dataset's license and annotation semantics; do not infer authenticity labels from document-type labels.
