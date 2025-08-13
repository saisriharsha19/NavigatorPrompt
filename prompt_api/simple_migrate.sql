-- simple_migrate.sql
-- Run this with: psql -d navigator_database -f simple_migrate.sql

-- Create users table
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

-- Create user_prompts table
CREATE TABLE IF NOT EXISTS user_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    prompt_text TEXT NOT NULL,
    task_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary VARCHAR,
    tags VARCHAR[]
);

-- Create library_submissions table
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

-- Add user_id to tasks table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tasks' AND column_name = 'user_id'
    ) THEN
        ALTER TABLE tasks ADD COLUMN user_id UUID;
    END IF;
END $$;

-- Update library_prompts table if it exists
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'library_prompts') THEN
        -- Add task_id column if it doesn't exist
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'library_prompts' AND column_name = 'task_id'
        ) THEN
            ALTER TABLE library_prompts ADD COLUMN task_id UUID;
        END IF;
        
        -- Add submission_id column if it doesn't exist
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'library_prompts' AND column_name = 'submission_id'
        ) THEN
            ALTER TABLE library_prompts ADD COLUMN submission_id UUID;
        END IF;
        
        -- Update user_id to UUID type if it's not already
        BEGIN
            ALTER TABLE library_prompts ALTER COLUMN user_id TYPE UUID USING user_id::UUID;
        EXCEPTION WHEN OTHERS THEN
            -- Column is already UUID type
        END;
    END IF;
END $$;

-- Update prompt_stars table if it exists
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'prompt_stars') THEN
        BEGIN
            ALTER TABLE prompt_stars ALTER COLUMN user_id TYPE UUID USING user_id::UUID;
        EXCEPTION WHEN OTHERS THEN
            -- Column is already UUID type
        END;
    END IF;
END $$;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_user_prompts_user_id ON user_prompts(user_id);
CREATE INDEX IF NOT EXISTS idx_library_submissions_user_id ON library_submissions(user_id);
CREATE INDEX IF NOT EXISTS idx_library_submissions_status ON library_submissions(status);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);

-- Create default admin user
INSERT INTO users (id, email, username, full_name, is_active, is_admin)
VALUES (gen_random_uuid(), 'admin@example.com', 'admin', 'System Administrator', true, true)
ON CONFLICT (email) DO NOTHING;

-- Create demo user
INSERT INTO users (id, email, username, full_name, is_active, is_admin)
VALUES (gen_random_uuid(), 'demo@example.com', 'demo_user', 'Demo User', true, false)
ON CONFLICT (email) DO NOTHING;

-- Display success message
SELECT 'Migration completed successfully!' as status;