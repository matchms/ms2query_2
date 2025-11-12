![GitHub License](https://img.shields.io/github/license/matchms/ms2query_2)


# MS2Query 2.0
more to come...

## Basic workflow (so far):

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



