# Data

The competition data is not redistributed here. To run the project, download
`kaggle_clustering_student_version.csv` from the competition's **Data** tab on Kaggle
(`consumption-disaggregation-2026`) and place it in this folder.

With the Kaggle API configured:

```python
import kagglehub
path = kagglehub.competition_download('consumption-disaggregation-2026')
```

`cache.npz` is created automatically on first load (a dense household × hour matrix).
