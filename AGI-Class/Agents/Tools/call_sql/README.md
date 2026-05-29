
# MySQL

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