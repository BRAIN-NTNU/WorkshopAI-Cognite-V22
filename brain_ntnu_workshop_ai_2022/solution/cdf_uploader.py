import glob
import logging
import os
from typing import List

from cognite.client import CogniteClient
from cognite.client.data_classes import FileMetadata
from cognite.extractorutils.uploader import FileUploadQueue
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load variables from .env file
load_dotenv()


def main() -> None:
    # Initialize the Cognite Client in order to communicate with Cognite Data Fusion
    # Ask someone from Cognite to provide you with the content of the .env file
    logger.info("Initializing CogniteClient")
    TENANT_ID = os.getenv("TENANT_ID")
    CDF_CLUSTER = os.getenv("CDF_CLUSTER")
    client = CogniteClient(
        token_url=f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
        token_client_id=os.getenv("CLIENT_ID"),
        token_client_secret=os.getenv("CLIENT_SECRET"),
        token_scopes=[f"https://{CDF_CLUSTER}.cognitedata.com/.default"],
        project=os.getenv("COGNITE_PROJECT"),
        base_url=f"https://{CDF_CLUSTER}.cognitedata.com",
        client_name="WorkshopAIClient_2022",
    )

    # Start the queue for uploading files
    upload_queue = FileUploadQueue(cdf_client=client, overwrite_existing=True)
    logger.info("Starting FileUploadQueue")
    upload_queue.start()

    # TODO (Task 6): Fill in your code ########################################################
    ##################################################################################
    # List all the image files in the data folder similar to this ["data/img1.png", "data/img2.png",...]
    files: List[str] = glob.glob("data/*.png")
    ##################################################################################

    # For each file add a FileMetadata object to the upload queue
    logger.info(f"Adding {files} to the upload queue")
    for file in files:
        file_name = file.split("/")[1]
        file_meta_data = FileMetadata(
            name=file_name,
            data_set_id=2300255353773196,
            mime_type="image/png",
            external_id=file_name,
            source="workshop_ai_drone",
        )
        upload_queue.add_to_upload_queue(file_meta_data, file)

    # Trigger an upload of the queue, clears queue afterwards
    logger.info(f"Uploading {len(files)} images to CDF")
    upload_queue.upload()
    logger.info("Done.")

    # Stop upload thread if running, and ensures that the upload queue is empty if ensure_upload is True.
    upload_queue.stop(ensure_upload=True)


if __name__ == "__main__":
    main()
