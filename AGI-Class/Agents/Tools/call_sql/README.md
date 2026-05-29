
# MySQL

The following instructions are based on implementing MySQL on MacOS.

## Installation

Open your macOS Terminal and run:

```bash
brew install mysql
```

## Start MySQL Service

Type the following CMD to run the service manually in the background:
```bash
mysql.server start
```

## Access MySQL Locally

Before we start to test MySql remotely, it is helpful to check it locally.

From terminal, type
```bash
mysql -u root -p
```
and type the password to access the MySQL service.

### Check Database

```
mysql> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| agi_class_test     |
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
5 rows in set (0.006 sec)
```

## Check Port

```
mysql> SHOW VARIABLES LIKE 'port';
+---------------+-------+
| Variable_name | Value |
+---------------+-------+
| port          | 3306  |
+---------------+-------+
1 row in set (0.008 sec)
```

### Check Tables

We can use 
```
mysql> USE agi_class_test;
```
to change database such that we can look up right tables.

```
mysql> SHOW TABLES;
+--------------------------+
| Tables_in_agi_class_test |
+--------------------------+
| Classes                  |
| Scores                   |
| Students                 |
+--------------------------+
```

Now we can run other queries:
```
mysql> select * from Scores;
+----------+------------+---------+-------+
| score_id | student_id | subject | score |
+----------+------------+---------+-------+
|        1 |          1 | 数学    |  85.5 |
|        2 |          1 | 英语    |    90 |
|        3 |          2 | 数学    |    78 |
|        4 |          3 | 英语    |  88.5 |
|        5 |          4 | 数学    |    92 |
+----------+------------+---------+-------+
5 rows in set (0.000 sec)
```

## Python Connector

We can simply run the following syntax to test Python connector to the MySQL service:
```Python
import pymysql
connection = pymysql.connect(
    host='127.0.0.1',
    port=3306,  # <-- Change this from 13306 to 3306
    user='root',
    password='hung123456',  # Make sure this matches what you set up
    database='agi_class_test',
    charset='utf8mb4'
)
cursor = connection.cursor()
```
