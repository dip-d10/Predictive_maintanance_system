import os
import time
from typing import List

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from src.entity.config_entity import AzureBlobConfig
from src.logger import logging


class AzureBlobClient:
    def __init__(self, config: AzureBlobConfig):
        load_dotenv()
        self.config = config
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.connection_string = os.getenv("AZURE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_CONTAINER_NAME", self.config.container_name)

        self.blob_service_client = None
        self.container_client = None

    def connect(self):
        if self.container_client is not None:
            return self.container_client

        if self.connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        elif self.account_name and self.account_key:
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(account_url=account_url, credential=self.account_key)
        else:
            raise ValueError("Azure Blob credentials are missing in environment variables")

        self.container_client = self.blob_service_client.get_container_client(self.container_name)
        if not self.container_client.exists():
            self.container_client.create_container()
            logging.info(f"Created Azure Blob container: {self.container_name}")

        return self.container_client

    def blob_exists(self, blob_name: str) -> bool:
        container = self.connect()
        blob_client = container.get_blob_client(blob=blob_name)
        return blob_client.exists()

    def upload_file(self, local_path: str, blob_name: str):
        container = self.connect()
        blob_client = container.get_blob_client(blob=blob_name)

        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    def download_file(self, blob_name: str, local_path: str):
        container = self.connect()
        blob_client = container.get_blob_client(blob=blob_name)

        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        with open(local_path, "wb") as file_data:
            download_stream = blob_client.download_blob()
            file_data.write(download_stream.readall())

    def delete_blob(self, blob_name: str):
        container = self.connect()
        blob_client = container.get_blob_client(blob=blob_name)
        if blob_client.exists():
            blob_client.delete_blob()

    def list_blobs(self, prefix: str = "") -> List[str]:
        container = self.connect()
        return [blob.name for blob in container.list_blobs(name_starts_with=prefix)]

    def move_blob(self, source: str, destination: str):
        container = self.connect()
        source_client = container.get_blob_client(blob=source)
        if not source_client.exists():
            raise FileNotFoundError(f"Source blob does not exist: {source}")

        destination_client = container.get_blob_client(blob=destination)
        copy_operation = destination_client.start_copy_from_url(source_client.url)

        copy_id = copy_operation.get("copy_id")
        for _ in range(30):
            destination_props = destination_client.get_blob_properties()
            copy_props = destination_props.copy
            if copy_props.status == "success":
                source_client.delete_blob()
                return
            if copy_props.status in ["aborted", "failed"]:
                raise RuntimeError(
                    f"Blob copy failed for {source} -> {destination} with status {copy_props.status}"
                )
            time.sleep(1)

        if copy_id:
            destination_client.abort_copy(copy_id)
        raise TimeoutError(f"Timed out moving blob {source} -> {destination}")

    def get_blob_uri(self, blob_name: str) -> str:
        self.connect()
        if self.account_name:
            return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"

        if self.blob_service_client:
            parsed_url = self.blob_service_client.url.rstrip("/")
            return f"{parsed_url}/{self.container_name}/{blob_name}"

        return f"{self.container_name}/{blob_name}"
