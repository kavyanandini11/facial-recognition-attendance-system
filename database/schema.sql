-- Facial Recognition Attendance System — MySQL schema
-- Run manually with:  mysql -u root -p < database/schema.sql
-- (`python main.py init-db` does the same thing automatically.)

CREATE DATABASE IF NOT EXISTS attendance_db;
USE attendance_db;

CREATE TABLE IF NOT EXISTS users (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    employee_code  VARCHAR(32)  NOT NULL UNIQUE,      -- e.g. id_014
    name           VARCHAR(120) NOT NULL,
    department     VARCHAR(80)  NULL,
    enrolled_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attendance (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT         NOT NULL,
    attendance_date  DATE        NOT NULL,
    check_in         TIME        NOT NULL,
    status           CHAR(1)     NOT NULL DEFAULT 'P', -- P = present, L = late
    confidence       FLOAT       NULL,                 -- LBPH distance (lower = better)
    created_at       TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_user_day (user_id, attendance_date), -- one record per person per day
    INDEX idx_date (attendance_date),
    CONSTRAINT fk_attendance_user FOREIGN KEY (user_id)
        REFERENCES users (id) ON DELETE CASCADE
);
