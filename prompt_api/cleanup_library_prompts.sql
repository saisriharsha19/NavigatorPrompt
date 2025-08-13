-- cleanup_library_prompts.sql
-- Clean up invalid user_id values in library_prompts table

-- First, let's see what we have
SELECT 'Current library_prompts data:' as info;
SELECT id, user_id, text, created_at FROM library_prompts;

-- Get a valid user ID from the users table
DO $$
DECLARE
    valid_user_id UUID;
BEGIN
    -- Get the demo user's ID
    SELECT id INTO valid_user_id FROM users WHERE email = 'demo@example.com' LIMIT 1;
    
    IF valid_user_id IS NULL THEN
        -- If no demo user, get any user
        SELECT id INTO valid_user_id FROM users LIMIT 1;
    END IF;
    
    IF valid_user_id IS NULL THEN
        -- If no users exist, create a demo user
        INSERT INTO users (id, email, username, full_name, is_active, is_admin)
        VALUES (gen_random_uuid(), 'demo@example.com', 'demo_user', 'Demo User', true, false)
        RETURNING id INTO valid_user_id;
    END IF;
    
    -- Update any invalid user_id values
    UPDATE library_prompts 
    SET user_id = valid_user_id 
    WHERE user_id IS NULL 
       OR NOT (user_id::text ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$');
    
    RAISE NOTICE 'Updated library_prompts with valid user_id: %', valid_user_id;
END $$;

-- Verify the cleanup
SELECT 'After cleanup:' as info;
SELECT id, user_id, text, created_at FROM library_prompts;