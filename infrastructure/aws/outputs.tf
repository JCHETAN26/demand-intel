output "url" { value = "http://${aws_lb.main.dns_name}" }
output "api_repository" { value = aws_ecr_repository.api.repository_url }
output "web_repository" { value = aws_ecr_repository.web.repository_url }
output "data_bucket" { value = aws_s3_bucket.data.bucket }
output "db_secret_arn" { value = aws_secretsmanager_secret.db.arn }
