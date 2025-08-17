# prompt_api/cleanup_duplicate_stars.py - Run this first
import asyncio
from sqlalchemy import text
from database import engine

async def cleanup_duplicate_stars():
    """Remove duplicate stars and add unique constraint"""
    async with engine.begin() as conn:
        try:
            print("Checking for duplicate stars...")
            
            # Find duplicates
            duplicate_check = await conn.execute(text("""
                SELECT prompt_id, user_id, COUNT(*) as count
                FROM prompt_stars 
                GROUP BY prompt_id, user_id 
                HAVING COUNT(*) > 1
                ORDER BY count DESC;
            """))
            
            duplicates = duplicate_check.fetchall()
            print(f"Found {len(duplicates)} duplicate combinations")
            
            for dup in duplicates:
                print(f"  Prompt {dup.prompt_id} + User {dup.user_id}: {dup.count} entries")
            
            if duplicates:
                print("\nRemoving duplicates...")
                
                # Remove duplicates, keeping only the first one
                await conn.execute(text("""
                    DELETE FROM prompt_stars 
                    WHERE id NOT IN (
                        SELECT DISTINCT ON (prompt_id, user_id) id 
                        FROM prompt_stars 
                        ORDER BY prompt_id, user_id, id
                    );
                """))
                
                print("✓ Duplicates removed")
            
            # Add unique constraint to prevent future duplicates
            print("Adding unique constraint...")
            try:
                await conn.execute(text("""
                    ALTER TABLE prompt_stars 
                    ADD CONSTRAINT unique_user_prompt_star 
                    UNIQUE (prompt_id, user_id);
                """))
                print("✓ Unique constraint added")
            except Exception as e:
                if "already exists" in str(e):
                    print("✓ Unique constraint already exists")
                else:
                    print(f"⚠️  Could not add unique constraint: {e}")
            
            # Verify cleanup
            final_check = await conn.execute(text("""
                SELECT prompt_id, user_id, COUNT(*) as count
                FROM prompt_stars 
                GROUP BY prompt_id, user_id 
                HAVING COUNT(*) > 1;
            """))
            
            remaining_duplicates = final_check.fetchall()
            if remaining_duplicates:
                print(f"⚠️  Still have {len(remaining_duplicates)} duplicates")
            else:
                print("✅ No duplicates remaining")
                
        except Exception as e:
            print(f"❌ Error cleaning up duplicates: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(cleanup_duplicate_stars())