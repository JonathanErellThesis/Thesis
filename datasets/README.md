# Datasets

Place the parquet files used by the experiments here:

- `100K_ratings.parquet`
- `1M_ratings.parquet`
- `10M_ratings.parquet`
- `Beauty-25-10_ratings.parquet`

Each file should contain at least the columns:

- `user_id`
- `item_id`
- `rating`

The loader expects the filename pattern `<dataset>_ratings.parquet`, where the
dataset key is passed to the scripts, for example `--dataset 1M`.
