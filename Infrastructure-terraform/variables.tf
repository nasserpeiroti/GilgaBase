variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region"
}

variable "cluster_name" {
  type        = string
  default     = "mlapp-eks"
}

variable "node_instance_types" {
  type        = list(string)
  default     = ["t3.small"]
}

variable "desired_size" {
  type        = number
  default     = 2
}

variable "app_image_tag" {
  type        = string
  # Example: 869508798872.dkr.ecr.us-east-1.amazonaws.com/mlapp:1.0.0
  description = "Full image URL (ECR) incl. tag"
}

variable "twelve_data_api_key" {
  type        = string
  description = "API key for Twelve Data"
  sensitive   = true
}

variable "project_name" {
  type        = string
  default     = "mlapp"
}
