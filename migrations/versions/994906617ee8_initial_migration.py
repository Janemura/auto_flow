"""Initial migration

Revision ID: 994906617ee8
Revises: 
Create Date: 2025-03-29 12:51:18.634474

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '994906617ee8'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('traffic_data')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('traffic_data',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('image_path', sa.VARCHAR(length=255), nullable=False),
    sa.Column('car_count', sa.INTEGER(), nullable=False),
    sa.Column('timestamp', sa.DATETIME(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###
