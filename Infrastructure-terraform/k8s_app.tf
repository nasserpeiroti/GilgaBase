locals {
  ns = var.project_name
}

resource "kubernetes_namespace" "app" {
  provider = kubernetes.eks
  metadata { name = local.ns }
}

resource "kubernetes_secret" "twelve_data" {
  provider = kubernetes.eks
  metadata {
    name      = "twelve-data-secret"
    namespace = kubernetes_namespace.app.metadata[0].name
  }
  data = {
    TWELVE_DATA_API_KEY = var.twelve_data_api_key
  }
  type = "Opaque"
}

# Deployment (2 replicas)
resource "kubernetes_deployment_v1" "mlapp" {
  provider = kubernetes.eks
  metadata {
    name      = "mlapp-deployment"
    namespace = kubernetes_namespace.app.metadata[0].name
    labels = {
      app = "mlapp"
    }
  }
  spec {
    replicas = 2
    selector {
      match_labels = { app = "mlapp" }
    }
    template {
      metadata {
        labels = { app = "mlapp" }
      }
      spec {
        container {
          name  = "mlapp"
          image = var.app_image_tag

          port {
            container_port = 8000
          }

          env {
            name = "TWELVE_DATA_API_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.twelve_data.metadata[0].name
                key  = "TWELVE_DATA_API_KEY"
              }
            }
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
          }

          liveness_probe {
            http_get {
              path = "/"
              port = 8000
            }
            initial_delay_seconds = 15
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }
      }
    }
  }
}

# Service (type LoadBalancer -> external DNS/hostname)
resource "kubernetes_service_v1" "mlapp" {
  provider = kubernetes.eks
  metadata {
    name      = "mlapp-service"
    namespace = kubernetes_namespace.app.metadata[0].name
    labels = { app = "mlapp" }
  }
  spec {
    type = "LoadBalancer"
    selector = { app = "mlapp" }
    port {
      port        = 80
      target_port = 8000
      protocol    = "TCP"
    }
  }
}
