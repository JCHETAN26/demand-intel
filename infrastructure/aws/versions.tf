terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.70" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  # Remote state: create the bucket once, then uncomment.
  # backend "s3" { bucket = "demand-intel-tfstate" key = "prod/terraform.tfstate" region = "us-west-2" }
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "demand-intel", managed_by = "terraform" } }
}
