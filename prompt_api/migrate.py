# migrate_to_v3.py
"""
Database migration script for upgrading to v3.0 with admin features.
Run this script to migrate your existing database schema.
"""

import asyncio
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from config import settings

async def migrate_database():
    """
    Migrate database schema to support admin features and user management.
    """
    engine = create_async_engine(settings.DATABASE_URL)
    
    print("Starting database migration to v3.0...")
    
    # Create users table
    async with engine.begin() as conn:
        try:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email VARCHAR UNIQUE NOT NULL,
                    username VARCHAR UNIQUE NOT NULL,
                    full_name VARCHAR,
                    is_active BOOLEAN DEFAULT true,
                    is_admin BOOLEAN DEFAULT false,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            print("✓ Created users table")
        except Exception as e:
            print(f"✓ Users table already exists: {e}")
    
    # Create user_prompts table
    async with engine.begin() as conn:
        try:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_prompts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    prompt_text TEXT NOT NULL,
                    task_type VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary VARCHAR,
                    tags VARCHAR[]
                );
            """))
            print("✓ Created user_prompts table")
        except Exception as e:
            print(f"✓ User_prompts table already exists: {e}")
    
    # Create library_submissions table
    async with engine.begin() as conn:
        try:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS library_submissions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                    prompt_text TEXT NOT NULL,
                    submission_notes TEXT,
                    status VARCHAR DEFAULT 'PENDING',
                    admin_notes TEXT,
                    reviewed_by UUID REFERENCES users(id),
                    reviewed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary VARCHAR,
                    tags VARCHAR[],
                    task_id UUID UNIQUE
                );
            """))
            print("✓ Created library_submissions table")
        except Exception as e:
            print(f"✓ Library_submissions table already exists: {e}")
    
    # Add user_id to tasks table
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE tasks ADD COLUMN user_id UUID;"))
            print("✓ Added user_id to tasks table")
        except Exception as e:
            print("✓ user_id column already exists in tasks table")
    
    # Update library_prompts table
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE library_prompts ADD COLUMN task_id UUID;"))
            print("✓ Added task_id to library_prompts table")
        except Exception:
            print("✓ task_id column already exists in library_prompts table")
        
        try:
            await conn.execute(text("ALTER TABLE library_prompts ADD COLUMN submission_id UUID;"))
            print("✓ Added submission_id to library_prompts table")
        except Exception:
            print("✓ submission_id column already exists in library_prompts table")
        
        try:
            await conn.execute(text("ALTER TABLE library_prompts ALTER COLUMN user_id TYPE UUID USING user_id::UUID;"))
            print("✓ Updated library_prompts user_id to UUID")
        except Exception:
            print("✓ library_prompts user_id already UUID type")
    
    # Update prompt_stars table
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE prompt_stars ALTER COLUMN user_id TYPE UUID USING user_id::UUID;"))
            print("✓ Updated prompt_stars user_id to UUID")
        except Exception:
            print("✓ prompt_stars user_id already UUID type")
    
    # Create indexes
    index_commands = [
        ("idx_users_email", "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);"),
        ("idx_users_username", "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);"),
        ("idx_user_prompts_user_id", "CREATE INDEX IF NOT EXISTS idx_user_prompts_user_id ON user_prompts(user_id);"),
        ("idx_library_submissions_user_id", "CREATE INDEX IF NOT EXISTS idx_library_submissions_user_id ON library_submissions(user_id);"),
        ("idx_library_submissions_status", "CREATE INDEX IF NOT EXISTS idx_library_submissions_status ON library_submissions(status);"),
        ("idx_tasks_user_id", "CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);")
    ]
    
    for index_name, cmd in index_commands:
        async with engine.begin() as conn:
            try:
                await conn.execute(text(cmd))
                print(f"✓ Created index {index_name}")
            except Exception as e:
                print(f"✓ Index {index_name} already exists")
    
    # Create default users
    async with engine.begin() as conn:
        try:
            admin_id = str(uuid.uuid4())
            await conn.execute(text("""
                INSERT INTO users (id, email, username, full_name, is_active, is_admin)
                VALUES (:id, 'admin@example.com', 'admin', 'System Administrator', true, true)
                ON CONFLICT (email) DO NOTHING;
            """), {"id": admin_id})
            print("✓ Created default admin user (admin@example.com)")
        except Exception as e:
            print(f"✓ Default admin user already exists: {e}")
        
        try:
            demo_id = str(uuid.uuid4())
            await conn.execute(text("""
                INSERT INTO users (id, email, username, full_name, is_active, is_admin)
                VALUES (:id, 'demo@example.com', 'demo_user', 'Demo User', true, false)
                ON CONFLICT (email) DO NOTHING;
            """), {"id": demo_id})
            print("✓ Created demo user (demo@example.com)")
        except Exception as e:
            print(f"✓ Demo user already exists: {e}")
    
    print("\n🎉 Database migration completed successfully!")
    print("\nDefault accounts created:")
    print("📧 Admin: admin@example.com (admin privileges)")
    print("📧 Demo: demo@example.com (regular user)")
    print(f"\n🔑 Admin Key: {settings.ADMIN_KEY}")
    print("\nNext steps:")
    print("1. Start your Celery worker: celery -A celery_worker.celery_app worker --loglevel=info")
    print("2. Start your FastAPI server: uvicorn main:app --reload")
    print("3. Access admin endpoints using X-Admin-Key header")

if __name__ == "__main__":
    asyncio.run(migrate_database())