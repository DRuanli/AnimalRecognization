# scripts/deploy.sh
#!/bin/bash

# Pull latest changes
git pull origin main

# Build and start Docker containers
docker-compose down
docker-compose build
docker-compose up -d

echo "Deployment completed successfully"