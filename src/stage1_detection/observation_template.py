"""
Step 1.2: Generate manual observation CSV template.
The user fills this in by visually inspecting each image.
No botanical knowledge required.
"""

import pandas as pd
from pathlib import Path
from src.utils.image_utils import get_image_files


def generate_observation_template(image_dir, output_csv):
    """
    Creates a CSV template for manual image observations.

    User should open this CSV and fill in each column
    by looking at each image. Only record what you SEE,
    no botanical knowledge needed.
    """
    image_files = get_image_files(image_dir)

    observations = []
    for img_file in image_files:
        observations.append(
            {
                "image_id": img_file.name,
                "file_path": str(img_file),
                "contains_flowers": "",
                "flower_count_approx": "",
                "has_multiple_flower_types": "",
                "contains_human": "",
                "contains_rocks_stones": "",
                "contains_other_objects": "",
                "visual_duplicate_of": "",
                "image_quality": "",
                "notes": "",
            }
        )

    df = pd.DataFrame(observations)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Template created for {len(df)} images")
    print(f"Saved to: {output_csv}")
    print(f"\n>>> NEXT STEP: Open {output_csv} and fill in observations manually <<<")
    print(f">>> Only record what you can SEE — no plant knowledge required <<<")

    return df


if __name__ == "__main__":
    from src.config import CONFIG

    generate_observation_template(
        CONFIG["paths"]["raw_images"],
        f"{CONFIG['paths']['csvs']}/01_human_observations.csv",
    )
