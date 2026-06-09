import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class PreprocessConfig:
    min_poi_freq: int = 10
    min_user_freq: int = 10
    session_time_interval_min: int = 60 * 24
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    remove_isolated_24h: bool = True
    ignore_singleton_trajectories: bool = True
    history_limit: int = 50
    train_min_history: int = 20
    region_decimals: int = 2


def read_tky_raw(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", header=None, engine="python", encoding="latin1")
    df = df[[0, 1, 2, 3, 4, 5, 6, 7]].copy()
    df.columns = [
        "UserId",
        "PoiId",
        "PoiCategoryCode",
        "PoiCategoryName",
        "Latitude",
        "Longitude",
        "TimezoneOffsetMinutes",
        "UTCTimeRaw",
    ]

    utc_dt = pd.to_datetime(
        df["UTCTimeRaw"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="raise",
    )
    local_dt = utc_dt + pd.to_timedelta(df["TimezoneOffsetMinutes"], unit="m")

    df["UTCTimeOffset"] = local_dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["LocalTime"] = pd.to_datetime(df["UTCTimeOffset"], format="%Y-%m-%d %H:%M:%S")
    df["Weekday"] = local_dt.dt.day_name()

    return df[
        [
            "UserId",
            "PoiId",
            "PoiCategoryCode",
            "PoiCategoryName",
            "Latitude",
            "Longitude",
            "TimezoneOffsetMinutes",
            "UTCTimeOffset",
            "LocalTime",
            "Weekday",
        ]
    ].sort_values(["UserId", "LocalTime"]).reset_index(drop=True)


def filter_low_frequency(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    poi_count = df.groupby("PoiId")["UserId"].count()
    keep_pois = set(poi_count[poi_count > cfg.min_poi_freq].index)
    df = df[df["PoiId"].isin(keep_pois)]

    user_count = df.groupby("UserId")["PoiId"].count()
    keep_users = set(user_count[user_count > cfg.min_user_freq].index)
    df = df[df["UserId"].isin(keep_users)]

    return df.sort_values(["UserId", "LocalTime"]).reset_index(drop=True)


def split_by_global_time(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    df = df.sort_values("LocalTime").reset_index(drop=True)
    n = len(df)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
    df["SplitTag"] = "train"
    df.loc[train_end:val_end - 1, "SplitTag"] = "validation"
    df.loc[val_end:, "SplitTag"] = "test"
    return df.sort_values(["UserId", "LocalTime"]).reset_index(drop=True)


def remove_isolated_checkins_24h(df: pd.DataFrame) -> pd.DataFrame:
    prev_t = df.groupby("UserId")["LocalTime"].shift(1)
    next_t = df.groupby("UserId")["LocalTime"].shift(-1)
    gap_prev = (df["LocalTime"] - prev_t).dt.total_seconds().fillna(float("inf"))
    gap_next = (next_t - df["LocalTime"]).dt.total_seconds().fillna(float("inf"))
    isolated = (gap_prev > 86400) & (gap_next > 86400)
    return df[~isolated].reset_index(drop=True)


def build_pseudo_sessions(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    diffs = df.groupby("UserId")["LocalTime"].diff()
    diffs_min = diffs.dt.total_seconds() / 60.0
    new_session = diffs.isna() | (diffs_min > cfg.session_time_interval_min)
    df["pseudo_session_trajectory_id"] = new_session.cumsum().astype(int) - 1
    return df


def ignore_singleton_sessions(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("pseudo_session_trajectory_id")["LocalTime"].transform("count")
    df.loc[counts == 1, "SplitTag"] = "ignore"
    return df


def remove_unseen_user_poi(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    train = df[df["SplitTag"] == "train"].copy()
    val = df[df["SplitTag"] == "validation"].copy()
    test = df[df["SplitTag"] == "test"].copy()

    train_users = set(train["UserId"])
    train_pois = set(train["PoiId"])

    val = val[val["UserId"].isin(train_users) & val["PoiId"].isin(train_pois)].reset_index(drop=True)
    test = test[test["UserId"].isin(train_users) & test["PoiId"].isin(train_pois)].reset_index(drop=True)

    sample = pd.concat([train, val, test], axis=0).sort_values(["UserId", "LocalTime"]).reset_index(drop=True)
    return {
        "sample": sample,
        "train_sample": train.reset_index(drop=True),
        "validate_sample": val,
        "test_sample": test,
    }


def add_sequence_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["UserId", "LocalTime"]).reset_index(drop=True)
    df["sequence_id"] = df.groupby("UserId").cumcount().astype("int64")
    return df


def build_id_map_from_train(train_series: pd.Series) -> Dict[object, int]:
    uniq = pd.unique(train_series)
    return {v: i + 1 for i, v in enumerate(uniq)}


def apply_map(series: pd.Series, mapping: Dict[object, int], default: int = 0) -> pd.Series:
    return series.map(mapping).fillna(default).astype("int64")


def encode_ids(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df[df["SplitTag"] == "train"].copy()
    uid_map = build_id_map_from_train(train_df["UserId"])
    pid_map = build_id_map_from_train(train_df["PoiId"])
    cat_map = build_id_map_from_train(train_df["PoiCategoryName"])

    df["UserId"] = apply_map(df["UserId"], uid_map)
    df["PoiId"] = apply_map(df["PoiId"], pid_map)
    df["PoiCategoryId"] = apply_map(df["PoiCategoryName"], cat_map)
    return df


def build_sample_tables(df: pd.DataFrame, cfg: PreprocessConfig, output_dir: str) -> None:
    train_users = set(df[df["SplitTag"] == "train"]["UserId"].unique())
    train_pois = set(df[df["SplitTag"] == "train"]["PoiId"].unique())
    df = df[df["UserId"].isin(train_users) & df["PoiId"].isin(train_pois)].copy()
    df = df.sort_values(["UserId", "UTCTimeOffset"]).reset_index(drop=True)

    results = []
    for user_id, user_df in df.groupby("UserId", sort=False):
        user_df = user_df.sort_values("UTCTimeOffset").reset_index(drop=True)
        grouped = user_df.groupby(["pseudo_session_trajectory_id", "SplitTag"], sort=False)
        for (traj_id, split_tag), traj_df in grouped:
            traj_df = traj_df.sort_values("UTCTimeOffset").reset_index(drop=True)
            start_time = traj_df["UTCTimeOffset"].iloc[0]
            history_df = user_df[user_df["UTCTimeOffset"] < start_time].copy()
            merged_df = pd.concat([history_df, traj_df], axis=0).sort_values("UTCTimeOffset")
            merged_df = merged_df.iloc[-cfg.history_limit:].reset_index(drop=True)

            if split_tag == "train" and len(merged_df) < cfg.train_min_history:
                continue

            results.append(
                {
                    "UserId": user_id,
                    "trajectory_id": traj_id,
                    "SplitTag": split_tag,
                    "history_count": len(history_df),
                    "current_count": len(traj_df),
                    "final_count": len(merged_df),
                    "sequence_PoiId": merged_df["PoiId"].tolist(),
                    "sequence_PoiCategoryId": merged_df["PoiCategoryId"].tolist(),
                    "sequence_PoiCategoryName": merged_df["PoiCategoryName"].tolist(),
                    "sequence_Latitude": merged_df["Latitude"].tolist(),
                    "sequence_Longitude": merged_df["Longitude"].tolist(),
                    "sequence_UTCTimeOffset": merged_df["UTCTimeOffset"].astype(str).tolist(),
                    "sequence_pseudo_session_trajectory_id": merged_df["pseudo_session_trajectory_id"].tolist(),
                }
            )

    result_df = pd.DataFrame(results)
    new_dir = os.path.join(output_dir, "new")
    os.makedirs(new_dir, exist_ok=True)

    train_df = result_df[result_df["SplitTag"] == "train"][["UserId", "sequence_PoiId", "sequence_UTCTimeOffset"]]
    val_df = result_df[result_df["SplitTag"] == "validation"][["UserId", "sequence_PoiId", "sequence_UTCTimeOffset"]]
    test_df = result_df[result_df["SplitTag"] == "test"][["UserId", "sequence_PoiId", "sequence_UTCTimeOffset"]]

    result_df.to_csv(os.path.join(output_dir, "trajectory_sequence_data.csv"), index=False)
    train_df.to_csv(os.path.join(new_dir, "train_poi_sequence.csv"), index=False)
    val_df.to_csv(os.path.join(new_dir, "validation_poi_sequence.csv"), index=False)
    test_df.to_csv(os.path.join(new_dir, "test_poi_sequence.csv"), index=False)


def build_region_ids(df: pd.DataFrame, decimals: int) -> pd.Series:
    lat_key = df["Latitude"].round(decimals).astype(str)
    lon_key = df["Longitude"].round(decimals).astype(str)
    region_key = lat_key + "_" + lon_key
    region_map = {value: idx + 1 for idx, value in enumerate(pd.unique(region_key))}
    return region_key.map(region_map).astype("int64")


def build_poi_info(train_df: pd.DataFrame, output_dir: str, cfg: PreprocessConfig) -> None:
    df = train_df[["PoiId", "PoiCategoryName", "Latitude", "Longitude", "UTCTimeOffset"]].copy()
    df["UTCTimeOffset"] = pd.to_datetime(df["UTCTimeOffset"], errors="coerce")
    df = df.dropna(subset=["UTCTimeOffset"]).reset_index(drop=True)
    df["region"] = build_region_ids(df, cfg.region_decimals)

    rows = []
    for pid, group in df.groupby("PoiId"):
        row0 = group.iloc[0]
        hour_counts = group["UTCTimeOffset"].dt.hour.value_counts().to_dict()
        visit_time_and_count = {
            int(hour): int(count) for hour, count in sorted(hour_counts.items(), key=lambda item: item[1], reverse=True)
        }
        rows.append(
            {
                "pid": int(pid),
                "category": row0["PoiCategoryName"],
                "region": int(row0["region"]),
                "latitude": float(row0["Latitude"]),
                "longitude": float(row0["Longitude"]),
                "visit_time_and_count": visit_time_and_count,
            }
        )

    poi_info = pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)
    poi_info.to_csv(os.path.join(output_dir, "poi_info.csv"), index=False)


def preprocess_tky(input_path: str, output_dir: str, cfg: Optional[PreprocessConfig] = None) -> None:
    cfg = cfg or PreprocessConfig()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sid"), exist_ok=True)

    df = read_tky_raw(input_path)
    print(f"After read: {df.shape}, users={df['UserId'].nunique()}, pois={df['PoiId'].nunique()}")

    df = filter_low_frequency(df, cfg)
    print(f"After filter: {df.shape}, users={df['UserId'].nunique()}, pois={df['PoiId'].nunique()}")

    df = split_by_global_time(df, cfg)
    print(df["SplitTag"].value_counts())

    if cfg.remove_isolated_24h:
        df = remove_isolated_checkins_24h(df)
        print(f"After removing isolated 24h: {df.shape}")

    df = build_pseudo_sessions(df, cfg)
    print(f"Sessions: {df['pseudo_session_trajectory_id'].nunique()}")

    if cfg.ignore_singleton_trajectories:
        df = ignore_singleton_sessions(df)
        print(f"Ignored rows: {(df['SplitTag'] == 'ignore').sum()}")

    result = remove_unseen_user_poi(df)
    final_df = add_sequence_id(result["sample"])
    final_df = encode_ids(final_df)
    final_df["UTCTimeOffset"] = pd.to_datetime(final_df["UTCTimeOffset"], format="%Y-%m-%d %H:%M:%S")

    final_df = final_df[
        [
            "UserId",
            "PoiId",
            "PoiCategoryId",
            "PoiCategoryCode",
            "PoiCategoryName",
            "Latitude",
            "Longitude",
            "UTCTimeOffset",
            "SplitTag",
            "pseudo_session_trajectory_id",
            "sequence_id",
        ]
    ].sort_values(["UserId", "UTCTimeOffset"]).reset_index(drop=True)

    sample_path = os.path.join(output_dir, "sample.csv")
    train_path = os.path.join(output_dir, "train_sample.csv")
    val_path = os.path.join(output_dir, "validate_sample.csv")
    test_path = os.path.join(output_dir, "test_sample.csv")

    final_df.to_csv(sample_path, index=False)
    final_df[final_df["SplitTag"] == "train"].to_csv(train_path, index=False)
    final_df[final_df["SplitTag"] == "validation"].to_csv(val_path, index=False)
    final_df[final_df["SplitTag"] == "test"].to_csv(test_path, index=False)

    build_sample_tables(final_df.copy(), cfg, output_dir)
    build_poi_info(final_df[final_df["SplitTag"] == "train"].copy(), output_dir, cfg)

    print("Saved files:")
    print(sample_path)
    print(train_path)
    print(val_path)
    print(test_path)
    print(os.path.join(output_dir, "poi_info.csv"))
    print(os.path.join(output_dir, "new", "train_poi_sequence.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GNPR-SID V2 dataset artifacts.")
    parser.add_argument("--dataset", choices=["tky"], required=True, help="Dataset name to preprocess.")
    parser.add_argument("--input-path", required=True, help="Path to the raw input file.")
    parser.add_argument("--output-dir", required=True, help="Directory where prepared files will be written.")
    parser.add_argument("--min-poi-freq", type=int, default=10)
    parser.add_argument("--min-user-freq", type=int, default=10)
    parser.add_argument("--session-gap-minutes", type=int, default=60 * 24)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--train-min-history", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        min_poi_freq=args.min_poi_freq,
        min_user_freq=args.min_user_freq,
        session_time_interval_min=args.session_gap_minutes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        history_limit=args.history_limit,
        train_min_history=args.train_min_history,
    )

    if args.dataset == "tky":
        preprocess_tky(args.input_path, args.output_dir, cfg)


if __name__ == "__main__":
    main()
