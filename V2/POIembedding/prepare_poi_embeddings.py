import argparse
import os

from category2vec import category2vec
from POI2emb import main as build_poi_embeddings


def run_pipeline(
    poi_info_csv: str,
    output_dir: str,
    category_model: str,
    category_dim: int,
    category_backend: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    category_dir = os.path.join(output_dir, "category")
    poi_dir = os.path.join(output_dir, "poi")
    os.makedirs(category_dir, exist_ok=True)
    os.makedirs(poi_dir, exist_ok=True)

    category2vec(
        csv_path=poi_info_csv,
        output_dir=category_dir,
        model_name=category_model,
        n_components=category_dim,
        category_column="category",
        backend=category_backend,
    )

    category_pkl = os.path.join(category_dir, "category_to_embedding.pkl")
    build_poi_embeddings(
        csv_path=poi_info_csv,
        category_pkl=category_pkl,
        cf_pkl="",
        output_dir=poi_dir,
    )

    print("Saved artifacts:")
    print(category_pkl)
    print(os.path.join(poi_dir, "poi_Emb_dict.pkl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare POI embedding artifacts for GNPR-SID V2.")
    parser.add_argument("--poi-info-csv", required=True, help="Path to poi_info.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for category and POI embedding artifacts")
    parser.add_argument("--category-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--category-dim", type=int, default=64, help="Category embedding dimension after PCA")
    parser.add_argument(
        "--category-backend",
        default="auto",
        choices=["auto", "sentence-transformers", "tfidf"],
        help="Category embedding backend. Use tfidf in offline environments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        poi_info_csv=args.poi_info_csv,
        output_dir=args.output_dir,
        category_model=args.category_model,
        category_dim=args.category_dim,
        category_backend=args.category_backend,
    )


if __name__ == "__main__":
    main()
