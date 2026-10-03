variable "region" {
  type    = string
  default = "us-west-2"
}
variable "name" {
  type    = string
  default = "demand-intel"
}
variable "api_image_tag" {
  type    = string
  default = "latest"
}
variable "web_image_tag" {
  type    = string
  default = "latest"
}
variable "api_cpu" {
  description = "Fargate CPU units per API task (spec §11.3 SLOs are stated for 1 vCPU)."
  type        = number
  default     = 1024
}
variable "api_memory" {
  type    = number
  default = 2048
}
variable "api_desired_count" {
  type    = number
  default = 2
}
variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}
variable "alarm_email" {
  description = "Optional address for CloudWatch alarm notifications."
  type        = string
  default     = ""
}
