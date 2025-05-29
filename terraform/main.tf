provider "aws" {
  region = var.aws_region
}

resource "aws_ecr_repository" "mlops_repo" {
  name = "mlops-rnn"
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecsTaskExecutionRole"
  assume_role_policy = data.aws_iam_policy_document.ecs_task.json
}

data "aws_iam_policy_document" "ecs_task" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_ecs_cluster" "mlops_cluster" {
  name = "mlops-cluster"
}

resource "aws_ecs_task_definition" "mlops_task" {
  family                   = "mlops-rnn-task"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  network_mode             = "awsvpc"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([{
    name      = "mlops-rnn"
    image     = "${aws_ecr_repository.mlops_repo.repository_url}:latest"
    essential = true
    portMappings = [{
      containerPort = 80
      hostPort      = 80
    }]
  }])
}

resource "aws_ecs_service" "mlops_service" {
  name            = "mlops-service"
  cluster         = aws_ecs_cluster.mlops_cluster.id
  task_definition = aws_ecs_task_definition.mlops_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [var.security_group_id]
    assign_public_ip = true
  }
}
# mwwa
resource "aws_s3_bucket" "dag_bucket" {
  bucket = var.dag_s3_bucket
}

resource "aws_mwaa_environment" "mlops_airflow" {
  name = "mlops-rnn-airflow"
  airflow_version = "2.7.2"
  execution_role_arn = var.execution_role_arn
  source_bucket_arn  = aws_s3_bucket.dag_bucket.arn

  dag_s3_path = "dags/"
  requirements_s3_path = "requirements.txt"

  network_configuration {
    subnet_ids         = var.subnet_ids
    security_group_ids = [var.security_group_id]
  }

  webserver_access_mode = "PUBLIC_ONLY"
}

