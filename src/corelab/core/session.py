import time

from sagemaker_core.helper.session_helper import Session, get_execution_role, s3_path_join
from sagemaker import image_uris
from sagemaker.session import Session as SageMakerSession


class CoreLabSession:

    def __init__(self, framework: str, project_name: str, default_folder: str | None = None, create_run_folder: bool = False):
        self.framework = framework
        self.project_name = project_name
        self.session_timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

        if create_run_folder:
            bucket_prefix = s3_path_join(default_folder if default_folder else "", self.session_timestamp)
        else:
            bucket_prefix = default_folder

        self.core_session = Session(default_bucket_prefix=bucket_prefix)
        self.role = get_execution_role()
        self.region = self.core_session.boto_region_name

    def print(self):
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
        return SageMakerSession(boto_session=self.core_session.boto_session)

    @property
    def base_s3_uri(self):
        """Get the base S3 URI with bucket and default prefix.

        Returns:
            str: Base S3 URI (s3://{bucket}/{default_bucket_prefix})
        """
        bucket = self.core_session.default_bucket()
        prefix = self.core_session.default_bucket_prefix

        return s3_path_join("s3://", bucket, prefix)

    @property
    def transform_output_s3_uri(self):
        return s3_path_join(self.base_s3_uri, "transform")

    def retrieve_image(self, version: str, instance_type: str = "ml.m5.xlarge"):
        image = image_uris.retrieve(
            framework=self.framework,
            region=self.region,
            version=version,
            py_version="py3",  # only for some frameworks
            instance_type=instance_type,
            sagemaker_session=self.core_session
        )
        return image

    @property
    def training_job_name(self):
        return '-'.join([self.framework, self.session_timestamp])

    @property
    def tuning_job_name(self):
        return '-'.join([self.framework, "tune", self.session_timestamp])

    @property
    def tuning_job_definition_name(self):
        return '-'.join([self.framework, "definition", self.session_timestamp])

    @property
    def model_name(self):
        return "-".join([self.project_name, self.framework, self.session_timestamp])

    @property
    def transform_job_name(self):
        return "-".join([self.project_name, self.framework, "prediction", self.session_timestamp])
