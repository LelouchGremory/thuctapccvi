"""001_initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-08-14

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    op.create_table(
        'employees',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('employee_code', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('department', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('employee_code')
    )

    op.create_table(
        'cameras',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('camera_code', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('source', sa.String(length=255), nullable=False),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('camera_code')
    )

    op.create_table(
        'face_profiles',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('employee_id', sa.String(length=36), nullable=False),
        sa.Column('embedding', Vector(512), nullable=False),
        sa.Column('image_path', sa.String(length=255), nullable=True),
        sa.Column('model_name', sa.String(length=50), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('active_ai_combo', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table(
        'recognition_events',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('camera_id', sa.String(length=36), nullable=True),
        sa.Column('employee_id', sa.String(length=36), nullable=True),
        sa.Column('track_id', sa.String(length=50), nullable=True),
        sa.Column('similarity', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('crop_image_path', sa.String(length=255), nullable=True),
        sa.Column('model_name', sa.String(length=50), nullable=True),
        sa.Column('active_ai_combo', sa.String(length=50), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id']),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id']),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table(
        'checkin_events',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('employee_id', sa.String(length=36), nullable=False),
        sa.Column('camera_id', sa.String(length=36), nullable=True),
        sa.Column('track_id', sa.String(length=50), nullable=True),
        sa.Column('checkin_time', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('image_path', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id']),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id']),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table(
        'system_backlog',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('track_id', sa.String(length=50), nullable=True),
        sa.Column('failure_stage', sa.String(length=50), nullable=False),
        sa.Column('failure_reason', sa.String(length=255), nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('active_ai_combo', sa.String(length=50), nullable=True),
        sa.Column('camera_id', sa.String(length=50), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('image_path', sa.String(length=255), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('system_backlog')
    op.drop_table('checkin_events')
    op.drop_table('recognition_events')
    op.drop_table('face_profiles')
    op.drop_table('cameras')
    op.drop_table('employees')
