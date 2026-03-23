# Employee Management System (Python + MySQL)

A simple command-line **Employee Management System** built in Python using a MySQL database.

This project allows you to:
- Add employees
- Remove employees
- Promote employees (increase salary)
- Display all employee records

The implementation is provided in a Jupyter Notebook:
- `employee_management_system.ipynb`

---

## ✨ Features

- **Add Employee** with ID, name, position, and salary
- **Remove Employee** by ID
- **Promote Employee** by updating salary with an increment amount
- **Display Employees** with all current records
- **Input-based menu system** for interactive use
- **Basic error handling** for SQL/database operations

---

## 🧰 Tech Stack

- **Python 3**
- **MySQL**
- **mysql-connector-python**
- **Jupyter Notebook**

---

## 📁 Project Structure

```text
39- Employee Management System/
├── employee_management_system.ipynb
└── README.md
```

---

## ✅ Prerequisites

Before running this project, ensure you have:

1. **Python 3.x** installed
2. **MySQL Server** installed and running
3. A MySQL user with access privileges
4. Jupyter Notebook or VS Code with Notebook support

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "39- Employee Management System"
```

2. Install required package:

```bash
pip install mysql-connector-python
```

---

## 🗄️ Database Setup

The notebook connects using:

```python
con = mysql.connector.connect(
    host="localhost", user="root", password="password", database="emp")
```

Create the database and table before running the functions:

```sql
CREATE DATABASE IF NOT EXISTS emp;
USE emp;

CREATE TABLE IF NOT EXISTS employees (
    id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position VARCHAR(100) NOT NULL,
    salary DECIMAL(10,2) NOT NULL
);
```

> **Important:** Update `user`, `password`, `host`, and `database` values in the notebook to match your local MySQL configuration.

---

## ▶️ How to Run

1. Open `employee_management_system.ipynb`
2. Run cells in order from top to bottom
3. After defining all functions, run:

```python
menu()
```

4. Use the on-screen menu:

- `1` → Add Employee
- `2` → Remove Employee
- `3` → Promote Employee
- `4` → Display Employees
- `5` → Exit

---

## 🧪 Example Workflow

1. Add employee with ID `101`
2. Promote employee `101` by salary increment
3. Display all employees to verify updates
4. Remove employee if needed

---

## 🚧 Possible Improvements

- Validate numeric salary input during add operation
- Add search employee by ID/name
- Add update employee details feature
- Add pagination for large employee lists
- Add logging and unit tests
- Convert notebook into modular Python scripts
- Add secure credential handling using environment variables

---

## 🔒 Security Note

Do not commit real database credentials. Prefer using environment variables or a `.env` file for sensitive values.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## 📄 License

This project is open-source and available under the **MIT License**.

---

## 🙌 Acknowledgements

Built as part of hands-on Python + MySQL learning projects.