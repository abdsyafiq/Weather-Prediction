from pymongo import MongoClient
from pymongo import errors
import mariadb

from typing import Optional
from typing import Tuple
import json

from logger import log


class MariaDBConnector:
    def __init__(self, username: str, password: str, host: str, database: str, table: str, location: str=None):
        self.user = username
        self.password = password
        self.host = host
        self.database = database
        self.table = table
        self.location = location
        self.conn = None
        self.cur = None

    def connect(self) -> None:
        try:
            # Connect to Specific Database MariaDB
            self.conn = mariadb.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                database=self.database,
            )
            self.cur = self.conn.cursor()
            log(f"Connected to MariaDB Server {self.host} Database {self.database}.")
        except mariadb.Error as e:
            log(f"Error: {e}")

    def fetch_data(self) -> Optional[Tuple[list, list]]:
        try:
            if not self.cur:
                self.connect()
            
            # Execute Query to Fetch Data
            if self.location:
                self.cur.execute(f"SELECT * FROM {self.table} WHERE location_name = %s", (self.location,))
            else:
                self.cur.execute(f"SELECT * FROM {self.table}")
            
            # Get Columns and Data
            columns = [col[0] for col in self.cur.description]
            rows = self.cur.fetchall()

            return columns, rows
        except mariadb.Error as e:
            log(f"Error: {e}")

    def close(self) -> None:
        try:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()
            log(f"Closed MariaDB Connection From Server {self.host} Database {self.database}.")
        except mariadb.Error as e:
            log(f"Error: {e}")


class MongoDBConnector:
    def __init__(self, username: str, password: str, server_1: str, server_2: str, server_3: str, port: str, auth_db: str, replica_set: str, database: str, collection: str):
        self.username = username
        self.password = password
        self.server_1 = server_1
        self.server_2 = server_2
        self.server_3 = server_3
        self.port = port
        self.auth_db = auth_db
        self.replica_set = replica_set
        self.database = database
        self.collection = collection
        self.client = None

    def connect(self) -> None:
        try:
            # Connect to MongoClient
            uri = (
                f"mongodb://{self.username}:{self.password}@"
                f"{self.server_1}:{self.port},{self.server_2}:{self.port},{self.server_3}:{self.port}/"
                f"{self.auth_db}?replicaSet={self.replica_set}"
            )
            self.client = MongoClient(uri)
            log(
                f"Connected to MongoDB Servers {self.server_1}, "
                f"{self.server_2}, and {self.server_3}. "
                f"In Replica Section {self.replica_set}."
            )
        except errors.ServerSelectionTimeoutError as e:
            log(f"Error: {e}")

    def fetch_data(self) -> json:
        try:
            db = self.client[self.database]
            collection = db[self.collection]

            # Find Data
            return collection.find().sort("_id", 1)

        except errors.ServerSelectionTimeoutError as e:
            log(f"Error: {e}")

    def close(self) -> None:
        try:
            if self.client:
                self.client.close()
                log(
                f"Closed MongoDB Connection From Servers {self.server_1}, "
                f"{self.server_2}, and {self.server_3}. "
                f"In Replica Section {self.replica_set}."
            )
        except errors.ServerSelectionTimeoutError as e:
            log(f"Error: {e}")
