import glob
import json
import os

from constants import JSON_CATEGORIES
from tqdm import tqdm


def create_annotation(annotation_type, source_dir, output_dir):
    combined_annotation = {
        "infos": {},
        "images": [],
        "annotations": [],
        "categories": JSON_CATEGORIES,
    }

    total_images = 0
    print(f"Creating {annotation_type} dataset...")
    for annotation_file in tqdm(train_annotations):
        # store json file in dictionary
        annotation = json.load(open(annotation_file))
        for index, image in enumerate(annotation["images"]):
            # copy image to output directory
            image_path = os.path.join(source_dir, image["img_path"])
            new_image_name = f"{total_images}.png"
            new_image_path = os.path.join(
                output_dir, annotation_type, "images", new_image_name
            )
            os.system(f"cp {image_path} {new_image_path}")

            image["img_path"] = new_image_path
            image["id"] = total_images
            combined_annotation["images"].append(image)

            image_annotation = annotation["annotations"][index]
            image_annotation["image_id"] = total_images
            image_annotation["id"] = total_images
            combined_annotation["annotations"].append(image_annotation)

            total_images += 1

    # save json file
    with open(
        os.path.join(output_dir, annotation_type, "annotations.json"), "w"
    ) as outfile:
        json.dump(combined_annotation, outfile)


if __name__ == "__main__":
    output_dir = "combined_dataset"
    source_dir = "combined_dataset_source"
    # get all files in root directory
    train_annotations = glob.glob(f"{source_dir}/**/train/annotations.json")
    test_annotations = glob.glob(f"{source_dir}/**/test/annotations.json")

    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, "train", "images"))
        os.makedirs(os.path.join(output_dir, "test", "images"))

    create_annotation("train", source_dir, output_dir)
    create_annotation("test", source_dir, output_dir)
