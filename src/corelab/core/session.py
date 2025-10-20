import os
import time
from types import SimpleNamespace

from sagemaker import image_uris
from sagemaker.session import Session as SageMakerSession
from sagemaker_core.helper.session_helper import Session, s3_path_join
from sagemaker_core.helper.session_helper import get_execution_role


class CoreLabSession:
    """Session manager for SageMaker Core lab experiments.

    This class provides a convenient wrapper around SageMaker Core sessions,
    managing AWS credentials, S3 paths, and resource naming conventions for
    machine learning experiments and deployments.

    Attributes:
        framework (str): The ML framework being used (e.g., 'xgboost', 'sklearn').
        project_name (str): Name of the project for resource naming.
        session_timestamp (str): ISO 8601 formatted timestamp for the session.
        core_session (Session): The underlying SageMaker Core session.
        role (str): The AWS IAM execution role ARN.
        region (str): AWS region where resources are deployed.
    """

    def __init__(self, framework: str, project_name: str, default_folder: str | None = None,
                 create_run_folder: bool = False, aws_profile: str = None):
        """Initialize a CoreLabSession.

        Args:
            framework: The ML framework to use (e.g., 'xgboost', 'sklearn').
            project_name: Name of the project for resource naming conventions.
            default_folder: Optional S3 folder prefix for all session outputs.
            create_run_folder: If True, creates a timestamped subfolder under default_folder.
            aws_profile: AWS profile name to use if execution role is not available.
        """
        self.framework = framework
        self.project_name = project_name
        self.session_timestamp = self._generate_timestamp()

        if create_run_folder:
            bucket_prefix = s3_path_join(default_folder if default_folder else "", self.session_timestamp)
        else:
            bucket_prefix = default_folder

        self.core_session = Session(default_bucket_prefix=bucket_prefix)

        try:
            role = get_execution_role()
            print("execution role available:", role)
        except Exception:
            print("falling back to profile:", aws_profile)
            os.environ['AWS_PROFILE'] = aws_profile
        self.role = get_execution_role()
        self.region = self.core_session.boto_region_name

    def print(self):
        """Print session configuration details.

        Displays AWS region, execution role, S3 bucket URI, framework, and project name.
        """
        print("AWS region:", self.region)
        print("Execution role", self.role)
        print("Output bucket uri:", self.base_s3_uri)
        print("Framework:", self.framework)
        print("Project name:", self.project_name)

    def get_sagemaker_session(self):
        """Create a SageMaker session from the boto3 session.

        Returns:
            sagemaker.Session: SageMaker session object
        """
        return SageMakerSession(boto_session=self.core_session.boto_session, default_bucket=self.core_session.default_bucket(), default_bucket_prefix=self.core_session.default_bucket_prefix)

    @property
    def base_s3_uri(self):
        """Get the base S3 URI with bucket and default prefix.

        Returns:
            str: Base S3 URI (s3://{bucket}/{default_bucket_prefix})
        """
        bucket = self.core_session.default_bucket()
        prefix = self.core_session.default_bucket_prefix

        return s3_path_join("s3://", bucket, prefix)


    def code_upload_location(self, folder_name: str):
        """Get S3 location details for code uploads.

        Args:
            folder_name: Name of the folder within the bucket prefix.

        Returns:
            SimpleNamespace: Object with properties:
                - bucket (str): S3 bucket name
                - prefix (str): Full S3 prefix path
                - s3_uri (str): Complete S3 URI
        """
        code_prefix = s3_path_join(self.core_session.default_bucket_prefix, folder_name)
        full_uri = s3_path_join(self.base_s3_uri, folder_name)
        obj = SimpleNamespace(bucket=self.core_session.default_bucket(), prefix=code_prefix, s3_uri=full_uri)
        return obj

    def upload_file(self, src_dir, src_file, dest_dir):
        """Upload a local file to S3.

        Args:
            src_dir: Local directory containing the source file.
            src_file: Name of the file to upload.
            dest_dir: Destination directory within the S3 bucket prefix.

        Returns:
            str: Full S3 URI of the uploaded file.
        """
        s3 = self.core_session.boto_session.client('s3')
        src_path = s3_path_join(src_dir, src_file)
        print("src_path:", src_path)
        dest_path = s3_path_join(self.core_session.default_bucket_prefix, dest_dir, src_file)
        print("dest_path:", dest_path)
        s3.upload_file(src_path, self.core_session.default_bucket(), dest_path)
        full_url = s3_path_join("s3://", self.core_session.default_bucket(), dest_path)
        print("full_url:", full_url)
        return full_url

    def update_timestamp(self):
        """Update the session timestamp to the current time.

        Regenerates the session timestamp, which affects all subsequent
        resource naming that includes the timestamp.
        """
        self.session_timestamp = self._generate_timestamp()

    @staticmethod
    def _generate_timestamp() -> str:
        """Generate an ISO 8601 formatted timestamp string.

        Returns:
            str: Timestamp in format 'YYYY-MM-DDTHH-MM-SS' (UTC).
        """
        return time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

    @property
    def transform_output_s3_uri(self):
        """Get the S3 URI for batch transform job outputs.

        Returns:
            str: S3 URI path for transform outputs.
        """
        return s3_path_join(self.base_s3_uri, "transform_output")

    @property
    def jobs_output_s3_uri(self):
        """Get the S3 URI for job outputs with project and timestamp.

        Returns:
            str: S3 URI path for job outputs in format {base}/{project}-{timestamp}/jobs.
        """
        return s3_path_join(self.base_s3_uri, '-'.join([self.project_name, self.session_timestamp]), "jobs")

    @property
    def pipeline_output_s3_uri(self):
        """Get the S3 URI for job outputs with project and timestamp.

        Returns:
            str: S3 URI path for job outputs in format {base}/{project}-{timestamp}/jobs.
        """
        return s3_path_join(self.base_s3_uri, "pipeline_output")

    def retrieve_image(self, version: str, instance_type: str = "ml.m5.xlarge"):
        """Retrieve the Docker image URI for the configured framework.

        Args:
            version: Framework version (e.g., '1.5-1' for XGBoost).
            instance_type: SageMaker instance type for image selection. Defaults to 'ml.m5.xlarge'.

        Returns:
            str: ECR image URI for the specified framework and version.
        """
        image = image_uris.retrieve(
            framework=self.framework,
            region=self.region,
            version=version,
            py_version="py3",  # only for some frameworks
            instance_type=instance_type,
            sagemaker_session=self.core_session,
        )
        return image

    @property
    def training_job_name(self):
        """Generate a unique name for a training job.

        Returns:
            str: Training job name in format {project}-{framework}-train-{timestamp}.
        """
        return '-'.join([self.project_name, self.framework, "train", self._generate_timestamp()])

    @property
    def tuning_job_name(self):
        """Generate a name for a hyperparameter tuning job.

        Returns:
            str: Tuning job name in format {project}-{framework}-tune-{session_timestamp}.
        """
        return '-'.join([self.project_name, self.framework, "tune", self.session_timestamp])

    @property
    def transform_job_name(self):
        """Generate a name for a batch transform job.

        Returns:
            str: Transform job name in format {project}-{framework}-prediction-{session_timestamp}.
        """
        return "-".join([self.project_name, self.framework, "prediction", self.session_timestamp])

    @property
    def processing_job_name(self):
        """Generate a unique name for a processing job.

        Returns:
            str: Processing job name in format {project}-{framework}-processing-{timestamp}.
        """
        return "-".join([self.project_name, self.framework, "processing", self._generate_timestamp()])

    @property
    def model_name(self):
        """Generate a name for a SageMaker model.

        Returns:
            str: Model name in format {project}-{framework}.
        """
        return "-".join([self.project_name, self.framework])

    @property
    def endpoint_config_name(self):
        """Generate a name for a standard endpoint configuration.

        Returns:
            str: Endpoint config name in format {project}-{framework}-endpoint-config.
        """
        return "-".join([self.project_name, self.framework, "endpoint-config"])

    @property
    def endpoint_name(self):
        """Generate a name for a standard endpoint.

        Returns:
            str: Endpoint name in format {project}-{framework}-endpoint.
        """
        return "-".join([self.project_name, self.framework, "endpoint"])

    @property
    def serverless_endpoint_config_name(self):
        """Generate a name for a serverless endpoint configuration.

        Returns:
            str: Serverless endpoint config name in format {project}-{framework}-serverless-config.
        """
        return "-".join([self.project_name, self.framework, "serverless-config"])

    @property
    def serverless_endpoint_name(self):
        """Generate a name for a serverless endpoint.

        Returns:
            str: Serverless endpoint name in format {project}-{framework}-serverless-endpoint.
        """
        return "-".join([self.project_name, self.framework, "serverless-endpoint"])
