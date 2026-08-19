#!/bin/bash
set -e

echo "Running Alembic Database Migrations..."
alembic upgrade head || echo "Database migration skipped or already applied."

echo "Starting Application Server..."
exec "$@"
