import docker
import os
import uuid
import logging
from typing import Optional, Dict
import time
import socket

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

MAX_CONTAINERS = 10
MODEL_FOLDER = 'models'  # This is good!


def wait_for_port(host: str, port: int, timeout: int = 60) -> bool:
    """Wait for a port to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect((host, port))
                logger.debug(f"Port {port} on {host} is open.")
                return True
        except (socket.timeout, ConnectionRefusedError) as e:
            logger.debug(f"Port {port} on {host} is not yet open: {e}")
            time.sleep(1)
    logger.error(f"Timeout waiting for port {port} on {host}.")
    return False




def get_running_containers() -> Dict[str, str]:
    """
    Получает список ID запущенных контейнеров инференса и их порты.
    Ключ - model_id, значение -  container_id
    """
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            # Correctly connect using DOCKER_HOST.
            docker_client = docker.DockerClient(base_url=os.environ.get('DOCKER_HOST'))
            return {
                container.attrs['Config']['Labels']['model_id']: container.id
                for container in docker_client.containers.list(filters={"label": "model_id", "status": "running"})
            }

        except docker.errors.DockerException as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Failed to connect to Docker daemon (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds. Error: {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Docker daemon after {max_retries} attempts. Error: {e}")
                raise
        except Exception as ex:
            logger.error(f"An unexpected error occurred: {ex}")
            raise


def start_inference_container(model_id: str, model_path: str) -> Optional[tuple[str, int]]:
    """Starts an inference container."""
    try:
        logger.debug(f"Attempting to start inference container for model {model_id}")
        docker_client = docker.DockerClient(base_url=os.environ.get('DOCKER_HOST'))
        container_name = f"inference-{model_id}-{uuid.uuid4().hex}"
        host_model_path = os.path.abspath(model_path)
        logger.debug(f"Host model path: {host_model_path}")

        logger.debug("Creating container...")
        container = docker_client.containers.run(
            image="inference-image:latest",
            name=container_name,
            mounts=[
                docker.types.Mount(
                    target="/app/model.joblib",
                    source=host_model_path,
                    type="bind",
                    read_only=True,
                )
            ],
            labels={"model_id": model_id},
            network="my_network",
            ports={"8010/tcp": None},
            detach=True,
        )
        logger.debug(f"Container created: {container.id}")

        container.reload()
        ports = container.ports
        logger.debug(f"Container ports: {ports}")

        host_port_list = ports.get('8010/tcp')
        logger.debug(f"{host_port_list}")
        if host_port_list:
            host_port = int(host_port_list[0]['HostPort'])
            logger.debug(f"Host port: {host_port}")
            if wait_for_port("localhost", host_port):
                logger.info(f"Inference container started: {container.id}, port: {host_port}")
                return container.id, host_port

        logger.error(f"Failed to start inference service for model {model_id}.")
        return None

    except docker.errors.ImageNotFound:
        logger.error("Inference image not found. Build it.")
        return None
    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        return None
    except Exception as ex:
        logger.error(f"Container didn't start: {ex}")
        return None



def stop_inference_container(container_id: str) -> None:
    """Stops an inference container."""
    docker_client = docker.DockerClient(base_url=os.environ.get('DOCKER_HOST'))
    try:
        container = docker_client.containers.get(container_id)
        container.stop()  # Stop
        container.remove()  # ...and remove.
        logger.info(f"Container stopped and removed: {container_id}")
    except docker.errors.NotFound:
        logger.warning(f"Container not found for stopping: {container_id}")
    except docker.errors.APIError as e:
        logger.error(f"Error stopping container {container_id}: {e}")
