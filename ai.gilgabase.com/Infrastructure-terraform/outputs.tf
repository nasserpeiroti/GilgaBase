output "ecr_repo_url" {
  value       = aws_ecr_repository.repo.repository_url
  description = "Push your Docker image here (use the tag you set in var.app_image_tag)."
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "kubectl_config_command" {
  value = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "service_dns" {
  # Will populate after Service is provisioned by AWS LB
  value       = try(kubernetes_service_v1.mlapp.status[0].load_balancer[0].ingress[0].hostname, "")
  description = "Public DNS of the LoadBalancer."
}
