#!/bin/bash

# Air Quality Streaming - Build and Deploy Script
# ================================================

echo "=== Air Quality Streaming - Build and Deploy ==="

# Set variables
IMAGE_NAME="air-quality-streaming"
CONTAINER_NAME="air-quality-streaming"

# Function to check if container is running
check_container_status() {
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        echo "✅ Container $CONTAINER_NAME is running"
        return 0
    else
        echo "❌ Container $CONTAINER_NAME is not running"
        return 1
    fi
}

# Function to show logs
show_logs() {
    echo "📋 Showing recent logs for $CONTAINER_NAME:"
    docker logs --tail=50 $CONTAINER_NAME
}

# Function to build image
build_image() {
    echo "🔨 Building Docker image..."
    docker build -t $IMAGE_NAME:latest .
    if [ $? -eq 0 ]; then
        echo "✅ Image built successfully"
    else
        echo "❌ Image build failed"
        exit 1
    fi
}

# Function to deploy with docker-compose
deploy_with_compose() {
    echo "🚀 Deploying with docker-compose..."
    
    # Stop existing container if running
    if check_container_status; then
        echo "🛑 Stopping existing container..."
        docker-compose down air-quality-streaming
    fi
    
    # Build and start
    docker-compose up -d --build air-quality-streaming
    
    if [ $? -eq 0 ]; then
        echo "✅ Deployment successful"
        
        # Wait a moment for container to start
        echo "⏳ Waiting for container to start..."
        sleep 10
        
        # Check status
        check_container_status
        
        # Show recent logs
        show_logs
    else
        echo "❌ Deployment failed"
        exit 1
    fi
}

# Function to deploy standalone (without docker-compose)
deploy_standalone() {
    echo "🚀 Deploying standalone container..."
    
    # Stop and remove existing container
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name $CONTAINER_NAME \
        --restart unless-stopped \
        --network airflow-docker_default \
        -e POSTGRES_HOST=postgres \
        -e POSTGRES_DB=airflow \
        -e POSTGRES_USER=airflow \
        -e POSTGRES_PASSWORD=airflow \
        -e POSTGRES_PORT=5432 \
        -e MINIO_HOST=minio:9000 \
        -e MINIO_ACCESS_KEY=admin \
        -e MINIO_SECRET_KEY=admin123 \
        -e MINIO_BUCKET=air-quality \
        -e WAQI_TOKEN=73f88f4b0cf15fdc4984cc221d78cff444761f42 \
        -e STREAMING_INTERVAL=300 \
        -v "$(pwd)/logs:/app/logs" \
        $IMAGE_NAME:latest
    
    if [ $? -eq 0 ]; then
        echo "✅ Standalone deployment successful"
        
        # Wait a moment for container to start
        echo "⏳ Waiting for container to start..."
        sleep 10
        
        # Check status
        check_container_status
        
        # Show recent logs
        show_logs
    else
        echo "❌ Standalone deployment failed"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo "📊 Current Status:"
    echo "=================="
    
    # Container status
    if check_container_status; then
        # Show container stats
        echo ""
        echo "📈 Container Stats:"
        docker stats --no-stream $CONTAINER_NAME
        
        echo ""
        echo "📋 Recent Logs (last 20 lines):"
        docker logs --tail=20 $CONTAINER_NAME
        
        echo ""
        echo "🔍 Container Details:"
        docker inspect $CONTAINER_NAME | grep -A 10 -B 5 '"Status":\|"Health":\|"RestartCount":'
    fi
}

# Function to stop service
stop_service() {
    echo "🛑 Stopping streaming service..."
    
    if check_container_status; then
        docker stop $CONTAINER_NAME
        echo "✅ Service stopped"
    else
        echo "ℹ️  Service was not running"
    fi
}

# Function to restart service  
restart_service() {
    echo "🔄 Restarting streaming service..."
    
    docker restart $CONTAINER_NAME
    
    if [ $? -eq 0 ]; then
        echo "✅ Service restarted"
        
        # Wait a moment
        echo "⏳ Waiting for service to start..."
        sleep 10
        
        # Check status
        check_container_status
        show_logs
    else
        echo "❌ Restart failed"
        exit 1
    fi
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up..."
    
    # Stop container
    docker stop $CONTAINER_NAME 2>/dev/null || true
    
    # Remove container
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Remove image
    docker rmi $IMAGE_NAME:latest 2>/dev/null || true
    
    echo "✅ Cleanup completed"
}

# Main menu
case "$1" in
    build)
        build_image
        ;;
    deploy)
        build_image
        deploy_with_compose
        ;;
    deploy-standalone)
        build_image
        deploy_standalone
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {build|deploy|deploy-standalone|status|logs|stop|restart|cleanup}"
        echo ""
        echo "Commands:"
        echo "  build             - Build Docker image"
        echo "  deploy            - Build and deploy with docker-compose"
        echo "  deploy-standalone - Build and deploy standalone container"
        echo "  status            - Show service status and stats"
        echo "  logs              - Show recent logs"
        echo "  stop              - Stop the service"
        echo "  restart           - Restart the service"
        echo "  cleanup           - Stop and remove container and image"
        echo ""
        echo "Examples:"
        echo "  $0 deploy          # Deploy with docker-compose (recommended)"
        echo "  $0 status          # Check service status"
        echo "  $0 logs            # View recent logs"
        echo "  $0 restart         # Restart if having issues"
        exit 1
        ;;
esac