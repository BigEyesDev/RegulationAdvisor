# AWS Deployment Reference

Region: **eu-central-1**  
ECS Cluster: **regulation-advisor**  
ECR Repository: **regulation-advisor**

> Replace `ACCOUNT_ID` with your 12-digit AWS account ID throughout.

---

## Prerequisites

```bash
# AWS CLI configured
aws configure   # region: eu-central-1, output: json

# Docker running locally (for build + push)
docker --version
```

---

## Push image to ECR

```bash
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# Create repository (one-time)
aws ecr create-repository --repository-name regulation-advisor --region eu-central-1

# Authenticate
aws ecr get-login-password --region eu-central-1 | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT}.dkr.ecr.eu-central-1.amazonaws.com

# Build and push
docker build -t regulation-advisor:latest .
docker tag regulation-advisor:latest \
  ${AWS_ACCOUNT}.dkr.ecr.eu-central-1.amazonaws.com/regulation-advisor:v0.5.0
docker push \
  ${AWS_ACCOUNT}.dkr.ecr.eu-central-1.amazonaws.com/regulation-advisor:v0.5.0
```

---

## Store secrets

```bash
aws secretsmanager create-secret \
  --name regulation-advisor/openrouter-api-key \
  --region eu-central-1 \
  --secret-string "${OPENROUTER_API_KEY}"

aws secretsmanager create-secret \
  --name regulation-advisor/tavily-api-key \
  --region eu-central-1 \
  --secret-string "${TAVILY_API_KEY}"
```

---

## IAM roles

Two roles are required. Create them in the AWS Console or with the CLI:

**ecsTaskExecutionRole** (managed policy: `AmazonECSTaskExecutionRolePolicy`) — allows ECS to pull
the image from ECR and write logs to CloudWatch.

**ecsTaskRole** — add an inline policy that allows `secretsmanager:GetSecretValue` on the two
secret ARNs above. The app reads API keys from Secrets Manager at startup.

---

## Register task definition

Edit `infra/task-definition.json`: replace both `ACCOUNT_ID` placeholders with your account ID.

```bash
aws ecs register-task-definition \
  --cli-input-json file://infra/task-definition.json \
  --region eu-central-1
```

---

## Create ECS cluster + service (see Feature 15 / Day 5)

```bash
# Cluster
aws ecs create-cluster --cluster-name regulation-advisor --region eu-central-1

# Service (run after ALB is created — substitute TARGET_GROUP_ARN and subnet/SG IDs)
aws ecs create-service \
  --cluster regulation-advisor \
  --service-name regulation-advisor-svc \
  --task-definition regulation-advisor \
  --desired-count 1 \
  --launch-type FARGATE \
  --region eu-central-1 \
  --network-configuration "awsvpcConfiguration={subnets=[SUBNET_ID],securityGroups=[SG_ID],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=TARGET_GROUP_ARN,containerName=regulation-advisor,containerPort=8000"
```

---

## CloudWatch logs

Logs stream to `/ecs/regulation-advisor` in CloudWatch.

```bash
aws logs tail /ecs/regulation-advisor --follow --region eu-central-1
```

---

## Cost control

Stop the service when not needed:
```bash
aws ecs update-service --cluster regulation-advisor \
  --service regulation-advisor-svc --desired-count 0 --region eu-central-1
```

Delete the ALB when done (biggest cost ~$18/month):
```bash
# Find ALB ARN
aws elbv2 describe-load-balancers --region eu-central-1
aws elbv2 delete-load-balancer --load-balancer-arn ALB_ARN --region eu-central-1
```
