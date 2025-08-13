-- migrate_saml.sql
-- Add SAML authentication fields to users table

-- Add new columns for SAML authentication and university data
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS first_name VARCHAR,
ADD COLUMN IF NOT EXISTS last_name VARCHAR,
ADD COLUMN IF NOT EXISTS student_id VARCHAR,
ADD COLUMN IF NOT EXISTS affiliation VARCHAR,
ADD COLUMN IF NOT EXISTS is_student BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS is_faculty BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS is_staff BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS saml_nameid VARCHAR UNIQUE,
ADD COLUMN IF NOT EXISTS last_login TIMESTAMP;

-- Create indexes for new fields
CREATE INDEX IF NOT EXISTS idx_users_student_id ON users(student_id);
CREATE INDEX IF NOT EXISTS idx_users_affiliation ON users(affiliation);
CREATE INDEX IF NOT EXISTS idx_users_saml_nameid ON users(saml_nameid);
CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login);
CREATE INDEX IF NOT EXISTS idx_users_is_student ON users(is_student);
CREATE INDEX IF NOT EXISTS idx_users_is_faculty ON users(is_faculty);
CREATE INDEX IF NOT EXISTS idx_users_is_staff ON users(is_staff);

-- Update existing users to set appropriate roles
UPDATE users SET is_student = true WHERE email LIKE '%@ufl.edu' AND is_admin = false;

-- Set first_name and last_name from full_name if they exist
UPDATE users 
SET 
    first_name = CASE 
        WHEN full_name IS NOT NULL AND position(' ' in full_name) > 0 
        THEN split_part(full_name, ' ', 1)
        ELSE full_name
    END,
    last_name = CASE 
        WHEN full_name IS NOT NULL AND position(' ' in full_name) > 0 
        THEN substring(full_name from position(' ' in full_name) + 1)
        ELSE NULL
    END
WHERE first_name IS NULL AND full_name IS NOT NULL;

SELECT 'SAML migration completed successfully!' as status;