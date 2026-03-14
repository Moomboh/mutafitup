rule fetch:
    params:
        url=lambda wc: config["datasets"]["schmirler_2024"]["url"],
        hash=lambda wc: config["datasets"]["schmirler_2024"]["hash"],
    output:
        tmp_zip="results/datasets_preprocess/schmirler_2024/raw/training_data.zip",
        tmp_extract_to=directory("results/datasets_preprocess/schmirler_2024/raw/training_data/"),
        train_data=directory("results/datasets_preprocess/schmirler_2024/pickle"),
    conda:
        "../../envs/prepare_data/fetch.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/schmirler_2024/fetch.log"
    script:
        "../../scripts/prepare_data/schmirler_2024/fetch.py"

checkpoint convert:
    input:
        pickle="results/datasets_preprocess/schmirler_2024/pickle",
    output:
        parquet=directory("results/datasets_preprocess/schmirler_2024/parquet"),
    params:
        schmirler_subsets=config["datasets"]["schmirler_2024"]["subsets"],
    conda:
        "../../envs/prepare_data/schmirler_2024/convert.yml"
    log:
        f"{LOG_PREFIX}/prepare_data/schmirler_2024/convert.log"
    script:
        "../../scripts/prepare_data/schmirler_2024/convert.py"
