variable "aws_region" {
  default = "us-east-1"
}

variable "subnet_ids" {
  type = list(string)
}

variable "security_group_id" {
  type = string
}

variable "dag_s3_bucket" {}
variable "execution_role_arn" {}

