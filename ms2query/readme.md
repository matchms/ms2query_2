## Basic workflow (so far):
This is for creating the database. This is not yet fully functional, so for now please use the notebooks/tutorial.ipynb if you already want to try out the prototype. 

### Library generation
```python
from ms2query.create_new_library import create_new_library

ms2query_lib = create_new_library(
    spectra_files=["spectra.mgf"],
    annotation_files=[],
    output_folder="my_ms2query_folder/",
    model_path="models/ms2deepscore.pt"
)
```

### Loading already generated library
```python
from ms2query.create_new_library import load_created_library

lib = load_created_library("my_ms2query_folder/")
```
