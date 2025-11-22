from neo4j import GraphDatabase
import sys
import os

uri = os.environ["NEO4J_URI"]
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

def create_new_database(driver, db_name):
    with driver.session(database="system") as session:
        try:
            session.run(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully.")
        except Exception as e:
            print(f"Error creating database '{db_name}': {e}")

if __name__ == '__main__':

    new_db_name = str(sys.argv[1])
    
    # Connect to the system database
    driver = GraphDatabase.driver(uri, auth=(username, password))
    create_new_database(driver, new_db_name)

    # Close the driver connection
    driver.close()
