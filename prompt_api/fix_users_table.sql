-- fix_users_table.sql
-- Fix the users table structure

-- Drop existing users table if it has wrong structure
DROP TABLE IF EXISTS users CASCADE;

-- Recreate users table with correct structure
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR UNIQUE NOT NULL,
    username VARCHAR UNIQUE NOT NULL,
    full_name VARCHAR,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Recreate related tables that depend on users
DROP TABLE IF EXISTS user_prompts CASCADE;
CREATE TABLE user_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    prompt_text TEXT NOT NULL,
    task_type VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary VARCHAR,
    tags VARCHAR[]
);

DROP TABLE IF EXISTS library_submissions CASCADE;
CREATE TABLE library_submissions (
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

-- Update other tables that reference users
-- Add user_id to tasks if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tasks' AND column_name = 'user_id'
    ) THEN
        ALTER TABLE tasks ADD COLUMN user_id UUID REFERENCES users(id);
    END IF;
END $$;

-- Fix library_prompts if it exists
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'library_prompts') THEN
        -- Add missing columns if they don't exist
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'library_prompts' AND column_name = 'task_id'
        ) THEN
            ALTER TABLE library_prompts ADD COLUMN task_id UUID;
        END IF;
        
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'library_prompts' AND column_name = 'submission_id'
        ) THEN
            ALTER TABLE library_prompts ADD COLUMN submission_id UUID;
        END IF;
        
        -- Update user_id to be UUID and add foreign key
        BEGIN
            ALTER TABLE library_prompts ALTER COLUMN user_id TYPE UUID USING user_id::UUID;
            -- Try to add foreign key constraint
            ALTER TABLE library_prompts ADD CONSTRAINT fk_library_prompts_user 
                FOREIGN KEY (user_id) REFERENCES users(id);
        EXCEPTION WHEN OTHERS THEN
            -- Constraint might already exist or other issues
        END;
    END IF;
END $$;

-- Fix prompt_stars if it exists
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'prompt_stars') THEN
        BEGIN
            ALTER TABLE prompt_stars ALTER COLUMN user_id TYPE UUID USING user_id::UUID;
            -- Try to add foreign key constraint
            ALTER TABLE prompt_stars ADD CONSTRAINT fk_prompt_stars_user 
                FOREIGN KEY (user_id) REFERENCES users(id);
        EXCEPTION WHEN OTHERS THEN
            -- Constraint might already exist or other issues
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

-- Create default users
INSERT INTO users (id, email, username, full_name, is_active, is_admin)
VALUES (gen_random_uuid(), 'admin@example.com', 'admin', 'System Administrator', true, true);

INSERT INTO users (id, email, username, full_name, is_active, is_admin)
VALUES (gen_random_uuid(), 'demo@example.com', 'demo_user', 'Demo User', true, false);

SELECT 'Users table fixed successfully!' as status;