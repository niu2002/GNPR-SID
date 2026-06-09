import os
import pandas as pd
import numpy as np
import argparse


def _build_tfidf_embeddings(categories):
    # Character ngrams are more robust for short venue categories and work offline.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        return vectorizer.fit_transform(categories).toarray()
    except Exception:
        # Fallback: deterministic character histogram with no external dependency.
        alphabet = sorted({ch for text in categories for ch in str(text).lower() if not ch.isspace()})
        if not alphabet:
            return np.zeros((len(categories), 1), dtype=np.float32)

        char_to_idx = {ch: idx for idx, ch in enumerate(alphabet)}
        matrix = np.zeros((len(categories), len(alphabet)), dtype=np.float32)
        for row_idx, text in enumerate(categories):
            for ch in str(text).lower():
                if ch in char_to_idx:
                    matrix[row_idx, char_to_idx[ch]] += 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums


def _build_sentence_transformer_embeddings(categories, model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(categories, show_progress_bar=True)


def _reduce_dimensions(embeddings, n_components):
    if n_components is None or n_components >= embeddings.shape[1]:
        return embeddings, None

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ vt[:n_components].T
    total_variance = np.sum(s ** 2)
    kept_variance = np.sum(s[:n_components] ** 2)
    explained_ratio = float(kept_variance / total_variance) if total_variance > 0 else 0.0
    return reduced, explained_ratio


def category2vec(
    csv_path,
    output_dir,
    model_name="all-MiniLM-L6-v2",
    n_components=None,
    category_column="category",
    backend="auto",
):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
    
    # Extract unique categories
    categories = df[category_column].unique().tolist()
    print(f"Found {len(categories)} unique categories")

    # Generate embeddings
    if backend not in {"auto", "sentence-transformers", "tfidf"}:
        raise ValueError(f"Unsupported backend: {backend}")

    used_backend = backend
    if backend == "tfidf":
        print("Generating category embeddings with offline TF-IDF backend...")
        embeddings = _build_tfidf_embeddings(categories)
    else:
        try:
            print(f"Loading sentence-transformer model: {model_name}")
            print("Generating category embeddings with sentence-transformers...")
            embeddings = _build_sentence_transformer_embeddings(categories, model_name)
            used_backend = "sentence-transformers"
        except Exception as exc:
            if backend == "sentence-transformers":
                raise
            print(f"Sentence-transformers unavailable, falling back to TF-IDF. Reason: {exc}")
            embeddings = _build_tfidf_embeddings(categories)
            used_backend = "tfidf"

    print(f"Original embedding shape: {embeddings.shape}")
    
    # Apply PCA if specified
    if n_components is not None and n_components < embeddings.shape[1]:
        print(f"Applying PCA to reduce dimensions to {n_components}")
        embeddings_reduced, explained_ratio = _reduce_dimensions(embeddings, n_components)
        print(f"Reduced embedding shape: {embeddings_reduced.shape}")
        if explained_ratio is not None:
            print(f"PCA explained variance ratio: {explained_ratio:.4f}")
        final_embeddings = embeddings_reduced
    else:
        final_embeddings = embeddings
    
    # Create mapping dictionary
    category_to_embedding = dict(zip(categories, final_embeddings))
    
    # Save results
    np.save(os.path.join(output_dir, "category_embeddings.npy"), final_embeddings)
    np.save(os.path.join(output_dir, "categories.npy"), np.array(categories, dtype=object))
    
    # Save mapping as dictionary (optional)
    import pickle
    with open(os.path.join(output_dir, "category_to_embedding.pkl"), 'wb') as f:
        pickle.dump(category_to_embedding, f)
    
    print(f"Results saved to: {output_dir}")
    print(f"- backend: {used_backend}")
    print(f"- category_embeddings.npy: Embedding matrix")
    print(f"- categories.npy: Category names")
    print(f"- category_to_embedding.pkl: Dictionary mapping")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate POI category embeddings")
    parser.add_argument("--csv_path", default=f"", help="Path to input CSV file")
    parser.add_argument("--output_dir", default=f"", help="Output directory for results")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--dim", type=int, default=64, help="Target dimensionality (PCA)")
    parser.add_argument("--column", default="category", help="Category column name")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "sentence-transformers", "tfidf"],
        help="Embedding backend. Use tfidf for offline environments.",
    )
    
    args = parser.parse_args()

    category2vec(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model,
        n_components=args.dim,
        category_column=args.column,
        backend=args.backend,
    )
