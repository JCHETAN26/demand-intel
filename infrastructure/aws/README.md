# AWS deployment

S3 + ECR + ECS/Fargate + RDS + CloudWatch, as in the plan. There is no Kubernetes; nothing
here needs it. GPUs aren't used: the API serves precomputed forecasts and a CPU event model.

```bash
cd infrastructure/aws
terraform init
terraform apply                                   # creates ~30 resources; costs money
aws s3 sync ../../data/serving s3://$(terraform output -raw data_bucket)/serving/
# build and push images (or run the "deploy" GitHub workflow)
aws ecr get-login-password | docker login --username AWS --password-stdin $(terraform output -raw api_repository | cut -d/ -f1)
docker build -f apps/api/Dockerfile -t $(terraform output -raw api_repository):latest ../..
docker build --build-arg NEXT_PUBLIC_API_URL=$(terraform output -raw url) -t $(terraform output -raw web_repository):latest ../../apps/web
docker push $(terraform output -raw api_repository):latest && docker push $(terraform output -raw web_repository):latest
aws ecs update-service --cluster demand-intel --service api --force-new-deployment
```

Rough monthly cost at the defaults (us-west-2, on demand): 2 API tasks (1 vCPU / 2 GB) +
1 web task + an ALB + db.t4g.micro is in the range of $90–130. Run `terraform destroy` when
you're done.
