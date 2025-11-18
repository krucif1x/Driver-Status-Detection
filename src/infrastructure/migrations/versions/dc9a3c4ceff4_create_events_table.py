"""Create events table

Revision ID: dc9a3c4ceff4
Revises: 
Create Date: 2025-10-16 05:28:02.048528

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dc9a3c4ceff4'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'events',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('timestamp', sa.DateTime, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('user_id', sa.Integer, nullable=False),
        sa.Column('event_type', sa.String, nullable=False),
        sa.Column('details', sa.Text)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('events')
