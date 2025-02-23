import sqlite3

# Connect to SQLite (creates fashion.db if it doesn't exist)
conn = sqlite3.connect("fashion.db")
cursor = conn.cursor()

# Create a table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS clothes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL
    )
''')

# Insert sample data
cursor.execute("INSERT INTO clothes (name, category, price) VALUES ('T-Shirt', 'Topwear', 15.99)")
conn.commit()

print("Database and table created successfully!")

conn.close()
