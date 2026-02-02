"""Docker Compose services lane for external dependencies.
from __future__ import annotations

This module provides functionality to manage external services like
PostgreSQL, Redis, MongoDB, etc. using Docker Compose for testing.
"""

import importlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServiceConfig:
    """Configuration for a Docker Compose service."""

    name: str
    image: str
    ports: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    command: str | None = None
    healthcheck: dict[str, Any] | None = None
    depends_on: list[str] = field(default_factory=list)

    def to_compose(self) -> dict[str, Any]:
        """Convert to Docker Compose format.

        Returns:
            Dictionary in Docker Compose service format.
        """
        service: dict[str, Any] = {
            "image": self.image,
        }

        if self.ports:
            service["ports"] = [f"{host}:{container}" for host, container in self.ports.items()]

        if self.environment:
            service["environment"] = self.environment

        if self.volumes:
            service["volumes"] = [f"{host}:{container}" for host, container in self.volumes.items()]

        if self.command:
            service["command"] = self.command

        if self.healthcheck:
            service["healthcheck"] = self.healthcheck

        if self.depends_on:
            service["depends_on"] = self.depends_on

        return service


class ServiceTemplates:
    """Pre-configured service templates for common databases and services."""

    @staticmethod
    def postgres(
        version: str = "15",
        port: int = 5432,
        user: str = "test",
        password: str = "test",
        database: str = "test",
    ) -> ServiceConfig:
        """Create PostgreSQL service configuration.

        Args:
            version: PostgreSQL version.
            port: Host port to expose.
            user: Database user.
            password: Database password.
            database: Database name.

        Returns:
            ServiceConfig for PostgreSQL.
        """
        return ServiceConfig(
            name="postgres",
            image=f"postgres:{version}",
            ports={f"{port}": "5432"},
            environment={
                "POSTGRES_USER": user,
                "POSTGRES_PASSWORD": password,
                "POSTGRES_DB": database,
            },
            healthcheck={
                "test": ["CMD-SHELL", "pg_isready -U test"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 5,
            },
        )

    @staticmethod
    def redis(
        version: str = "7",
        port: int = 6379,
    ) -> ServiceConfig:
        """Create Redis service configuration.

        Args:
            version: Redis version.
            port: Host port to expose.

        Returns:
            ServiceConfig for Redis.
        """
        return ServiceConfig(
            name="redis",
            image=f"redis:{version}",
            ports={f"{port}": "6379"},
            command="redis-server --appendonly yes",
            healthcheck={
                "test": ["CMD", "redis-cli", "ping"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 5,
            },
        )

    @staticmethod
    def mysql(
        version: str = "8.0",
        port: int = 3306,
        user: str = "test",
        password: str = "test",
        database: str = "test",
    ) -> ServiceConfig:
        """Create MySQL service configuration.

        Args:
            version: MySQL version.
            port: Host port to expose.
            user: Database user.
            password: Database password.
            database: Database name.

        Returns:
            ServiceConfig for MySQL.
        """
        return ServiceConfig(
            name="mysql",
            image=f"mysql:{version}",
            ports={f"{port}": "3306"},
            environment={
                "MYSQL_ROOT_PASSWORD": "root",
                "MYSQL_DATABASE": database,
                "MYSQL_USER": user,
                "MYSQL_PASSWORD": password,
            },
            healthcheck={
                "test": ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "test", "-ptest"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 5,
            },
        )

    @staticmethod
    def mongodb(
        version: str = "6.0",
        port: int = 27017,
    ) -> ServiceConfig:
        """Create MongoDB service configuration.

        Args:
            version: MongoDB version.
            port: Host port to expose.

        Returns:
            ServiceConfig for MongoDB.
        """
        return ServiceConfig(
            name="mongodb",
            image=f"mongo:{version}",
            ports={f"{port}": "27017"},
            environment={
                "MONGO_INITDB_ROOT_USERNAME": "test",
                "MONGO_INITDB_ROOT_PASSWORD": "test",
            },
            healthcheck={
                "test": ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"],
                "interval": "5s",
                "timeout": "5s",
                "retries": 5,
            },
        )

    @staticmethod
    def elasticsearch(
        version: str = "8.0",
        port: int = 9200,
    ) -> ServiceConfig:
        """Create Elasticsearch service configuration.

        Args:
            version: Elasticsearch version.
            port: Host port to expose.

        Returns:
            ServiceConfig for Elasticsearch.
        """
        return ServiceConfig(
            name="elasticsearch",
            image=f"elasticsearch:{version}",
            ports={f"{port}": "9200"},
            environment={
                "discovery.type": "single-node",
                "ES_JAVA_OPTS": "-Xms512m -Xmx512m",
                "xpack.security.enabled": "false",
            },
            healthcheck={
                "test": ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"],
                "interval": "10s",
                "timeout": "10s",
                "retries": 10,
            },
        )

    @staticmethod
    def rabbitmq(
        version: str = "3-management",
        port: int = 5672,
        management_port: int = 15672,
    ) -> ServiceConfig:
        """Create RabbitMQ service configuration.

        Args:
            version: RabbitMQ version.
            port: AMQP port to expose.
            management_port: Management UI port to expose.

        Returns:
            ServiceConfig for RabbitMQ.
        """
        return ServiceConfig(
            name="rabbitmq",
            image=f"rabbitmq:{version}",
            ports={
                f"{port}": "5672",
                f"{management_port}": "15672",
            },
            environment={
                "RABBITMQ_DEFAULT_USER": "test",
                "RABBITMQ_DEFAULT_PASS": "test",
            },
            healthcheck={
                "test": ["CMD", "rabbitmq-diagnostics", "ping"],
                "interval": "10s",
                "timeout": "5s",
                "retries": 5,
            },
        )


class DockerComposeManager:
    """Manages Docker Compose services for testing."""

    def __init__(
        self,
        work_dir: str,
        project_name: str = "rfsn-test",
        time_mode: str = "live",
    ):
        """Initialize the Docker Compose manager.

        Args:
            work_dir: Directory for compose files.
            project_name: Docker Compose project name.
        """
        self.work_dir = work_dir
        self.project_name = project_name
        self.time_mode = time_mode
        self.compose_file = os.path.join(work_dir, "docker-compose.yml")
        self.services: list[ServiceConfig] = []
        self._running = False

    def add_service(self, service: ServiceConfig) -> None:
        """Add a service to the composition.

        Args:
            service: Service configuration to add.
        """
        # Check for duplicate service names
        for existing in self.services:
            if existing.name == service.name:
                raise ValueError(f"Service '{service.name}' already exists")

        self.services.append(service)

    def add_postgres(
        self,
        version: str = "15",
        port: int = 5432,
        user: str = "test",
        password: str = "test",
        database: str = "test",
    ) -> None:
        """Add PostgreSQL service.

        Args:
            version: PostgreSQL version.
            port: Host port to expose.
            user: Database user.
            password: Database password.
            database: Database name.
        """
        self.add_service(
            ServiceTemplates.postgres(
                version=version,
                port=port,
                user=user,
                password=password,
                database=database,
            )
        )

    def add_redis(
        self,
        version: str = "7",
        port: int = 6379,
    ) -> None:
        """Add Redis service.

        Args:
            version: Redis version.
            port: Host port to expose.
        """
        self.add_service(ServiceTemplates.redis(version=version, port=port))

    def add_mysql(
        self,
        version: str = "8.0",
        port: int = 3306,
        user: str = "test",
        password: str = "test",
        database: str = "test",
    ) -> None:
        """Add MySQL service.

        Args:
            version: MySQL version.
            port: Host port to expose.
            user: Database user.
            password: Database password.
            database: Database name.
        """
        self.add_service(
            ServiceTemplates.mysql(
                version=version,
                port=port,
                user=user,
                password=password,
                database=database,
            )
        )

    def add_mongodb(
        self,
        version: str = "6.0",
        port: int = 27017,
    ) -> None:
        """Add MongoDB service.

        Args:
            version: MongoDB version.
            port: Host port to expose.
        """
        self.add_service(ServiceTemplates.mongodb(version=version, port=port))

    def add_elasticsearch(
        self,
        version: str = "8.0",
        port: int = 9200,
    ) -> None:
        """Add Elasticsearch service.

        Args:
            version: Elasticsearch version.
            port: Host port to expose.
        """
        self.add_service(ServiceTemplates.elasticsearch(version=version, port=port))

    def add_rabbitmq(
        self,
        version: str = "3-management",
        port: int = 5672,
        management_port: int = 15672,
    ) -> None:
        """Add RabbitMQ service.

        Args:
            version: RabbitMQ version.
            port: AMQP port to expose.
            management_port: Management UI port to expose.
        """
        self.add_service(
            ServiceTemplates.rabbitmq(
                version=version,
                port=port,
                management_port=management_port,
            )
        )

    def _generate_compose_file(self) -> str:
        """Generate Docker Compose YAML content.

        Returns:
            YAML content for docker-compose.yml.
        """
        compose: dict[str, Any] = {
            "version": "3.8",
            "services": {},
        }

        for service in self.services:
            compose["services"][service.name] = service.to_compose()

        # Convert to YAML
        yaml = importlib.import_module("yaml")
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)

    def write_compose_file(self) -> None:
        """Write docker-compose.yml to work directory."""
        os.makedirs(self.work_dir, exist_ok=True)

        compose_content = self._generate_compose_file()

        with open(self.compose_file, "w") as f:
            f.write(compose_content)

    def up(self, detached: bool = True) -> dict[str, Any]:
        """Start services using Docker Compose.

        Args:
            detached: Run in detached mode.

        Returns:
            Result dictionary with status and output.
        """
        if not self.services:
            return {"ok": True, "message": "No services to start"}

        self.write_compose_file()

        cmd = ["docker-compose", "-f", self.compose_file, "-p", self.project_name, "up", "-d"]

        if detached:
            cmd.append("-d")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=60, check=False,
            )

            self._running = result.returncode == 0

            return {
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": "Timeout starting services",
                "exit_code": -1,
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "error": "docker-compose not found",
                "exit_code": -1,
            }

    def down(self, volumes: bool = False) -> dict[str, Any]:
        """Stop and remove services.

        Args:
            volumes: Remove volumes as well.

        Returns:
            Result dictionary with status and output.
        """
        if not self._running:
            return {"ok": True, "message": "Services not running"}

        cmd = ["docker-compose", "-f", self.compose_file, "-p", self.project_name, "down"]

        if volumes:
            cmd.append("-v")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=60, check=False,
            )

            self._running = False

            return {
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": "Timeout stopping services",
                "exit_code": -1,
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "error": "docker-compose not found",
                "exit_code": -1,
            }

    def logs(self, service: str | None = None, tail: int = 100) -> dict[str, Any]:
        """Get logs from services.

        Args:
            service: Specific service name (None for all).
            tail: Number of lines to show.

        Returns:
            Result dictionary with logs.
        """
        cmd = [
            "docker-compose",
            "-f",
            self.compose_file,
            "-p",
            self.project_name,
            "logs",
            "--tail",
            str(tail),
        ]

        if service:
            cmd.append(service)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=30, check=False,
            )

            return {
                "ok": result.returncode == 0,
                "logs": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": "Timeout getting logs",
            }

    def ps(self) -> dict[str, Any]:
        """List running services.

        Returns:
            Result dictionary with service status.
        """
        cmd = ["docker-compose", "-f", self.compose_file, "-p", self.project_name, "ps"]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=30, check=False,
            )

            return {
                "ok": result.returncode == 0,
                "output": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": "Timeout listing services",
            }

    def wait_for_healthy(self, timeout: int = 60) -> dict[str, Any]:
        """Wait for all services to become healthy.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Result dictionary with status.
        """
        if (self.time_mode or "").lower() != "live":
            raise RuntimeError("wait_for_healthy requires time_mode='live' (uses wall-clock sleep)")
        poll_interval_sec = 2
        max_attempts = max(1, int(timeout // poll_interval_sec))

        for _ in range(max_attempts):
            all_healthy = True

            for service in self.services:
                if service.healthcheck:
                    # Check service health using docker inspect
                    cmd = [
                        "docker",
                        "inspect",
                        "--format",
                        "{{.State.Health.Status}}",
                        f"{self.project_name}_{service.name}_1",
                    ]

                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=5, check=False,
                        )

                        if result.returncode != 0 or result.stdout.strip() != "healthy":
                            all_healthy = False
                            break
                    except Exception:
                        all_healthy = False
                        break

            if all_healthy:
                return {"ok": True, "message": "All services healthy"}

            time.sleep(poll_interval_sec)

        return {
            "ok": False,
            "error": f"Timeout waiting for services ({timeout}s)",
        }

    def get_service_env(self, service_name: str) -> dict[str, str]:
        """Get environment variables for a service.

        Args:
            service_name: Name of the service.

        Returns:
            Dictionary of environment variables.
        """
        for service in self.services:
            if service.name == service_name:
                return service.environment.copy()

        return {}

    def get_service_url(
        self, service_name: str, default_port: int | None = None
    ) -> str | None:
        """Get connection URL for a service.

        Args:
            service_name: Name of the service.
            default_port: Default port if not found.

        Returns:
            Connection URL or None.
        """
        for service in self.services:
            if service.name == service_name:
                # Get first port mapping
                if service.ports:
                    host_port = list(service.ports.keys())[0]
                    return f"localhost:{host_port}"

        if default_port:
            return f"localhost:{default_port}"

        return None

    def cleanup(self) -> None:
        """Clean up compose file and directory."""
        if os.path.exists(self.compose_file):
            os.remove(self.compose_file)

        # Only remove work directory if it's empty
        try:
            if os.path.isdir(self.work_dir) and not os.listdir(self.work_dir):
                os.rmdir(self.work_dir)
        except Exception:
            pass


def detect_required_services(repo_dir: str) -> list[str]:
    """Detect required services from repository configuration.

    Args:
        repo_dir: Path to repository.

    Returns:
        List of service names (postgres, redis, mysql, mongodb, etc.).
    """
    services = []

    # Check for common service indicators
    files_to_check = [
        (
            "requirements.txt",
            ["psycopg2", "sqlalchemy", "django.db.backends.postgresql"],
            "postgres",
        ),
        ("requirements.txt", ["redis", "django-redis", "celery[redis]"], "redis"),
        ("requirements.txt", ["pymysql", "mysqlclient", "django.db.backends.mysql"], "mysql"),
        ("requirements.txt", ["pymongo", "motor"], "mongodb"),
        ("requirements.txt", ["elasticsearch", "elasticsearch-dsl"], "elasticsearch"),
        ("requirements.txt", ["pika", "kombu"], "rabbitmq"),
        ("package.json", ["pg", "postgres"], "postgres"),
        ("package.json", ["redis"], "redis"),
        ("package.json", ["mysql", "mysql2"], "mysql"),
        ("package.json", ["mongodb", "mongoose"], "mongodb"),
        ("go.mod", ["postgres", "pgx", "lib/pq"], "postgres"),
        ("go.mod", ["redis", "go-redis"], "redis"),
        ("go.mod", ["mongo", "mongo-driver"], "mongodb"),
        ("Cargo.toml", ["postgres", "tokio-postgres"], "postgres"),
        ("Cargo.toml", ["redis", "redis-rs"], "redis"),
        ("Cargo.toml", ["mongo", "mongodb"], "mongodb"),
    ]

    for filename, indicators, service in files_to_check:
        file_path = os.path.join(repo_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    content = f.read().lower()

                for indicator in indicators:
                    if indicator in content:
                        if service not in services:
                            services.append(service)
                        break
            except Exception:
                pass

    return services


def create_services_manager(repo_dir: str, services: list[str]) -> DockerComposeManager:
    """Create a Docker Compose manager with detected services.

    Args:
        repo_dir: Path to repository.
        services: List of service names to add.

    Returns:
        DockerComposeManager instance.
    """
    # Create work directory
    work_dir = os.path.join(repo_dir, ".rfsn-services")

    manager = DockerComposeManager(work_dir=work_dir)

    # Add services
    for service in services:
        if service == "postgres":
            manager.add_postgres()
        elif service == "redis":
            manager.add_redis()
        elif service == "mysql":
            manager.add_mysql()
        elif service == "mongodb":
            manager.add_mongodb()
        elif service == "elasticsearch":
            manager.add_elasticsearch()
        elif service == "rabbitmq":
            manager.add_rabbitmq()

    return manager
