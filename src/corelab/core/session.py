import os
import time
from types import SimpleNamespace

from sagemaker import image_uris
from sagemaker.session import Session as SageMakerSession
from sagemaker_core.helper.session_helper import Session, s3_path_join
from sagemaker_core.helper.session_helper import get_execution_role


class CoreLabSession:

    def __init__(self, framework: str, project_name: str, default_folder: str | None = None, create_run_folder: bool = False, aws_profile: str = None):
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
        except Exception as e:
            print("falling back to profile:", aws_profile)
            os.environ['AWS_PROFILE'] = aws_profile
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
    def training_code_upload(self):
        """
            Returns: an object with two properties 'bucket' and 'prefix' for code uploading
        """
        code_prefix = self.core_session.default_bucket_prefix.rstrip('/') + '/code/train'
        obj = SimpleNamespace(bucket=self.core_session.default_bucket(), prefix=code_prefix)
        return obj

    @property
    def inference_code_upload(self):
        """
            Returns: an object with two properties 'bucket' and 'prefix' for code uploading
        """
        code_prefix = self.core_session.default_bucket_prefix.rstrip('/') + '/code/infer'
        obj = SimpleNamespace(bucket=self.core_session.default_bucket(), prefix=code_prefix)
        return obj

    def update_timestamp(self):
        self.session_timestamp = self._generate_timestamp()

    @staticmethod
    def _generate_timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

    @property
    def transform_output_s3_uri(self):
        return s3_path_join(self.base_s3_uri, "transform")

    @property
    def jobs_output_s3_uri(self):
        return s3_path_join(self.base_s3_uri, "jobs")

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
        return '-'.join([self.framework, self._generate_timestamp()])

    @property
    def tuning_job_name(self):
        return '-'.join([self.framework, "tune", self.session_timestamp])

    @property
    def transform_job_name(self):
        return "-".join([self.project_name, self.framework, "prediction", self.session_timestamp])

    @property
    def model_name(self):
        return "-".join([self.project_name, self.framework])

    @property
    def endpoint_config_name(self):
        return "-".join([self.project_name, self.framework, "endpoint-config"])

    @property
    def endpoint_name(self):
        return "-".join([self.project_name, self.framework, "endpoint"])

    @property
    def serverless_endpoint_config_name(self):
        return "-".join([self.project_name, self.framework, "serverless-config"])

    @property
    def serverless_endpoint_name(self):
        return "-".join([self.project_name, self.framework, "serverless-endpoint"])
